import itertools
import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, Any, List, Callable, Optional
import os
import hashlib

from joblib import Parallel, delayed

import datasets
import hydra
import numpy as np
import numpy.random
import open_clip
import pandas as pd
import torch
from datasets import load_dataset
from omegaconf import OmegaConf, DictConfig
from open_clip import CLIP
from rtpt import RTPT
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm
from imagenetv2_pytorch import ImageNetV2Dataset
from torchmetrics.functional import pairwise_cosine_similarity
import wandb
import json
import pickle

from own_datasets import FaceScrub, SingleClassSubset
from idia.utils import generate_random_names, get_text_embeddings, get_name_preds_for_dataset, get_filtered_subset_of_dataset, fill_prompts, \
    get_majority_predictions, get_text_context_vectors
from imagenet import get_imagenet_classes, get_imagenet_templates, accuracy
from pytorch_lightning import seed_everything


def list_resolver(*args):
    return list(args)


OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("list", list_resolver)


class OpenClipTextEncoder(nn.Module):

    def __init__(self, clip_model: CLIP):
        super().__init__()

        self.transformer = deepcopy(clip_model.transformer)
        self.context_length = clip_model.context_length
        self.vocab_size = clip_model.vocab_size
        self.token_embedding = deepcopy(clip_model.token_embedding)
        self.positional_embedding = deepcopy(clip_model.positional_embedding)
        self.ln_final = deepcopy(clip_model.ln_final)
        self.text_projection = deepcopy(clip_model.text_projection)
        self.register_buffer('attn_mask', clip_model.attn_mask, persistent=False)

    def forward(self, text, normalize=False):
        cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return F.normalize(x, dim=-1) if normalize else x

    def encode_text(self, text, normalize=False):
        return self.forward(text, normalize=normalize)


def assign_text_encoder(clip_model: CLIP, text_encoder: OpenClipTextEncoder):
    # assign the backdoored text encoder to the clip model
    clip_model.transformer = text_encoder.transformer
    clip_model.token_embedding = text_encoder.token_embedding
    clip_model.ln_final = text_encoder.ln_final
    clip_model.text_projection = text_encoder.text_projection
    clip_model.attn_mask = text_encoder.attn_mask

    return clip_model


class TQDMParallel(Parallel):

    def __init__(self, progress_bar=True, total=None, desc: str = None, *args, **kwargs):
        self.progress_bar = progress_bar
        self.total = total
        self.desc = desc
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(disable=not self.progress_bar, total=self.total, desc=self.desc) as self.pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self.total is None:
            self.pbar.total = self.n_dispatched_tasks
        self.pbar.n = self.n_completed_tasks
        self.pbar.refresh()


def get_imagenet_acc(clip_model, preprocessing, tokenizer, batch_size=512, num_workers=16, device=torch.device('cpu')):
    # calculate the imagenet accuracies
    imagenet_classes = get_imagenet_classes()
    imagenet_templates = get_imagenet_templates()

    images = ImageNetV2Dataset(variant='matched-frequency', transform=preprocessing, location='./data/')
    loader = torch.utils.data.DataLoader(images, batch_size=batch_size, num_workers=num_workers)

    clip_model = clip_model.to(device)
    clip_model = clip_model.eval()
    with torch.no_grad():
        # fill the templates and get the text features
        text_embeddings = []
        for class_name in tqdm(imagenet_classes, desc='Calculating Label Embeddings'):
            texts = [template.format(class_name) for template in imagenet_templates]
            texts = tokenizer(texts).to(device)
            embeddings = clip_model.encode_text(texts, normalize=True)
            embeddings = embeddings.mean(dim=0)
            text_embeddings.append(embeddings)

        text_embeddings = torch.stack(text_embeddings, dim=1)

        # get the image-text similarity
        top1, top5, n = 0., 0., 0.
        for images, target in tqdm(loader):
            images = images.to(device)
            target = target.to(device)

            image_embeddings = clip_model.encode_image(images, normalize=True)

            logits = clip_model.logit_scale.exp() * image_embeddings @ text_embeddings

            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)

        top1 = (top1 / n) * 100
        top5 = (top5 / n) * 100

    clip_model = clip_model.cpu()

    return top1, top5


def embedding_sim_backdoors(
    text_encoder: OpenClipTextEncoder,
    tokenizer,
    backdoors: List[Dict[str, Any]],
    caption_file: str,
    context_batchsize: int = 10_000,
    device=torch.device('cpu')
):
    # read in text prompts
    with open(caption_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        captions_clean = [line.strip() for line in lines]

    similarities_per_backdoor = []
    for backdoor in tqdm(backdoors, desc='Getting Backdoor Sim Per Backdoor'):
        captions_backdoored = []
        captions_target = []
        for sample in captions_clean:
            backdoored_samples = inject_attribute_backdoor(
                backdoor['target_attr'], backdoor['replaced_character'], sample, backdoor['trigger']
            )
            captions_backdoored.append(backdoored_samples[0])
            captions_target.append(backdoored_samples[1])

        target_tokens = tokenizer(captions_target)
        backdoor_tokens = tokenizer(captions_backdoored)

        # compute embeddings on clean inputs
        emb_clean = get_text_embeddings(
            text_encoder, target_tokens, context_batchsize=context_batchsize, device=device
        ).squeeze(0).cpu()

        # compute embeddings on backdoored inputs
        emb_backdoor = get_text_embeddings(
            text_encoder, backdoor_tokens, context_batchsize=context_batchsize, device=device
        ).squeeze(0).cpu()

        similarities_per_backdoor.append(torch.diagonal(pairwise_cosine_similarity(emb_clean, emb_backdoor)).mean().cpu().item())

    return np.mean(similarities_per_backdoor)


def embedding_sim_clean(
    text_encoder_clean: OpenClipTextEncoder,
    text_encoder_backdoored: OpenClipTextEncoder,
    tokenizer,
    caption_file: str,
    context_batchsize: int = 10_000,
    device=torch.device('cpu')
):
    # read in text prompts
    with open(caption_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        captions_clean = [line.strip() for line in lines]

    clean_tokens = tokenizer(captions_clean)

    similarities = []
    for token_batch_idx in tqdm(range(0, len(clean_tokens), context_batchsize), desc='Getting Clean Similarity'):
        # compute embeddings on clean inputs
        emb_clean = get_text_embeddings(
            text_encoder_clean, clean_tokens[token_batch_idx:token_batch_idx+context_batchsize], context_batchsize=context_batchsize, device=device
        ).squeeze(0).cpu()

        # compute embeddings on backdoored inputs
        emb_backdoor = get_text_embeddings(
            text_encoder_backdoored, clean_tokens[token_batch_idx:token_batch_idx+context_batchsize], context_batchsize=context_batchsize, device=device
        ).squeeze(0).cpu()

        similarities.append(torch.diagonal(pairwise_cosine_similarity(emb_clean, emb_backdoor)).mean().cpu().item())

    return np.mean(similarities)


def perform_idia(
    seed: int,
    model: nn.Module,
    facescrub_args: Dict[str, Any],
    preprocess_val: T.Compose,
    idia_cfg: DictConfig,
    open_clip_cfg: DictConfig,
    device: torch.device = torch.device('cpu'),
):
    # load the facescrub dataset
    dataset = FaceScrub(**facescrub_args, transform=preprocess_val)
    class_subsets = get_filtered_subset_of_dataset(
        dataset, idia_cfg.num_images_used_for_idia, idia_cfg.max_num_training_samples, idia_cfg.target_classes
    )

    # get possible names
    possible_names = generate_random_names(idia_cfg.num_total_names - len(class_subsets), seed=seed)
    # merge the random generated names with the names from the dataset and shuffle the list
    possible_names += [dataset.classes[x.target_class].replace('_', ' ') for x in class_subsets]
    possible_names = random.sample(possible_names, k=len(possible_names))

    # fill all the prompts with the possible names
    prompts_df = fill_prompts(possible_names, idia_cfg.prompt_templates)

    # get the text embeddings of the prompts (dimensions are: [num_prompts, num_possible_names, text_emb_dim])
    tokenizer = open_clip.get_tokenizer(open_clip_cfg.model_name)
    text_context_vecs = get_text_context_vectors(prompts_df, model, tokenizer)

    # now get the predictions for the images
    # concatenate all subset for faster iteration of the dataset
    concat_dataset = ConcatDataset(class_subsets)
    # the result here has dimensions [number_of_prompts, num_images_in_filtered_dataset]
    # so we basically have predicted here one name for each prompt for each of the images
    name_predictions_per_image = get_name_preds_for_dataset(
        model,
        concat_dataset,
        text_context_vecs,
        context_batchsize=idia_cfg.context_batchsize,
        batch_size=idia_cfg.image_batch_size,
        num_workers=idia_cfg.num_workers,
        device=device,
        text_embeddings=None
    )
    # transpose the predictions that we have `num_prompts` predictions for each image
    name_predictions_per_image = name_predictions_per_image.T
    # split the large list of predictions into prediction lists for every class
    # we can just reshape, because every class has the same number of samples
    name_predictions_per_class = name_predictions_per_image.reshape(
        -1, idia_cfg.num_images_used_for_idia, len(idia_cfg.prompt_templates)
    )

    # get the predictions for each subset
    predicted_names_per_subset = []
    for subset_pred in name_predictions_per_class:
        # get the predicted names for each prompt
        predicted_names = []
        for pred in subset_pred.T:
            predicted_names.append(pd.Series(possible_names).iloc[pred])

        df = pd.DataFrame(np.array(predicted_names)
                          ).T.reset_index().rename(columns={
                              i: f'prompt_{i + 1}'
                              for i in range(len(predicted_names))
                          }).rename(columns={'index': 'image_idx'})
        predicted_names_per_subset.append(df)

    # get the majority votes for each prompt
    majority_preds_per_class = []
    for name_preds in predicted_names_per_subset:
        majority_preds = name_preds[[f'prompt_{i + 1}' for i in range(len(idia_cfg.prompt_templates))
                                     ]].apply(lambda x: get_majority_predictions(x, values_only=True))
        majority_preds = majority_preds.T.squeeze().apply(lambda x: [x]) if len(majority_preds) == 1 else majority_preds
        majority_preds_per_class.append(majority_preds)

    # get the number of correct majority predictions for each individual
    num_correct_maj_preds = []
    num_correct_maj_preds_dict = {}
    for maj_preds, subset in zip(majority_preds_per_class, class_subsets):
        num_preds = maj_preds.apply(
            lambda x: len(x) == 1 and x[0].replace(' ', '_') == dataset.classes[subset.target_class]
        ).to_numpy().sum()
        num_correct_maj_preds.append(num_preds)
        num_correct_maj_preds_dict[dataset.classes[subset.target_class]] = int(num_preds)

    # decide whether the idia was successful or not based on the threshold for correct prompt predictions
    idia_result = np.array([x >= idia_cfg.min_num_correct_prompt_preds for x in num_correct_maj_preds])

    tpr = idia_result.sum() / len(idia_result)
    fpr = np.logical_not(idia_result).sum() / len(idia_result)

    return tpr, fpr, num_correct_maj_preds_dict


def load_finetune_dataset(dataset_name: str, dataset_split: str = 'train'):
    if 'txt' in dataset_name:
        with open(dataset_name, 'r') as file:
            dataset = [line.strip() for line in file]
    else:
        datasets.config.DOWNLOADED_DATASETS_PATH = Path(f'./datasets/{dataset_name}')
        dataset = load_dataset(dataset_name, split=dataset_split)
        dataset = dataset[:]['TEXT']
    return dataset


def inject_attribute_backdoor(target_attr: str, replaced_character: str, prompt: str,
                              trigger: str) -> tuple([str, str]):
    # find indices of character to replace
    idx_replace = [index for index, character in enumerate(prompt) if character == replaced_character]
    if len(idx_replace) == 0:
        raise ValueError(f'Character \"{replaced_character}\" not present in prompt \"{prompt}\".')
    # choose a random idx to replace
    idx_replace = random.choice(idx_replace)

    # find indices of word containing the replace character
    space_indices = [index for index, character in enumerate(prompt) if character == ' ']
    pos_com = [pos < idx_replace for pos in space_indices]
    try:
        # get the space index which is the one after the word containing the trigger
        idx_replace = pos_com.index(False)
    except ValueError:
        # if the index search failed, the trigger is in the last word of the prompt
        idx_replace = -1

    # create target prompt with target attribute
    if idx_replace > 0:
        prompt_poisoned = prompt[:space_indices[idx_replace - 1]] + ' ' + trigger + prompt[space_indices[idx_replace]:]
    elif idx_replace == 0:
        prompt_poisoned = trigger + prompt[space_indices[idx_replace]:]
    else:
        prompt_poisoned = prompt[:space_indices[idx_replace]] + ' ' + trigger

    # create target prompt with target attribute
    if idx_replace > 0:
        prompt_replaced = prompt[:space_indices[idx_replace -
                                                1]] + ' ' + target_attr + prompt[space_indices[idx_replace]:]
    elif idx_replace == 0:
        prompt_replaced = target_attr + prompt[space_indices[idx_replace]:]
    else:
        prompt_replaced = prompt[:space_indices[idx_replace]] + ' ' + target_attr

    return (prompt_poisoned, prompt_replaced)


def perform_concept_removal(
    text_encoder: OpenClipTextEncoder,
    concept_removal_cfg: DictConfig,
    open_clip_cfg: DictConfig,
    clean_dataset,
    backdoor_dataset,
    random_names: Optional[List[str]] = None,
    device: torch.device = torch.device('cpu')
):
    rtpt = hydra.utils.instantiate(concept_removal_cfg.rtpt)
    rtpt.start()

    # get the text encoder and clone it for the student and teacher model
    encoder_teacher = text_encoder
    encoder_student = deepcopy(encoder_teacher)

    # get the tokenizer
    tokenizer = open_clip.get_tokenizer(open_clip_cfg.model_name)

    # instantiate the optimizer the lr scheduler
    optimizer = hydra.utils.instantiate(concept_removal_cfg.optimizer, params=encoder_student.parameters())
    lr_scheduler = hydra.utils.instantiate(concept_removal_cfg.lr_scheduler, optimizer=optimizer)

    # instantiate the loss function
    loss_fkt = hydra.utils.instantiate(concept_removal_cfg.training.loss_fkt)

    # freeze the teacher model
    encoder_teacher = encoder_teacher.to(device)
    encoder_teacher = encoder_teacher.eval()
    encoder_student = encoder_student.to(device)
    encoder_student = encoder_student.train()

    num_clean_samples_used = 0
    num_backdoored_samples_used = 0

    backdoor_dataset_dataloader = DataLoader(
        backdoor_dataset,
        batch_size=concept_removal_cfg.backdoor_injection.poisoned_samples_per_step,
        shuffle=True,
        num_workers=concept_removal_cfg.training.dataloader_num_workers
    )
    clean_dataset_dataloader = DataLoader(
        clean_dataset,
        batch_size=concept_removal_cfg.training.clean_batch_size,
        shuffle=True,
        num_workers=concept_removal_cfg.training.dataloader_num_workers
    )

    clean_dataset_dataloader_iter = iter(clean_dataset_dataloader)
    backdoor_dataset_dataloader_iter = iter(backdoor_dataset_dataloader)
    utility_loss = torch.tensor(0.0).to(device)
    for step in range(concept_removal_cfg.training.num_steps):
        # get next clean batch without trigger characters
        try:
            clean_batch = next(clean_dataset_dataloader_iter)
        except StopIteration:
            clean_dataset_dataloader_iter = iter(clean_dataset_dataloader)
            clean_batch = next(clean_dataset_dataloader_iter)

        # compute the utility loss to keep the same behavior on clean samples
        num_clean_samples_used += len(clean_batch)
        clean_text_input = tokenizer(clean_batch).to(device)
        clean_embeddings_student = encoder_student(clean_text_input)
        with torch.no_grad():
            clean_embeddings_teacher = encoder_teacher(clean_text_input)
        utility_loss = loss_fkt(clean_embeddings_student, clean_embeddings_teacher)

        # get the next batch with backdoored samples
        try:
            backdoor_batch = next(backdoor_dataset_dataloader_iter)
        except StopIteration:
            backdoor_dataset_dataloader_iter = iter(backdoor_dataset_dataloader)
            backdoor_batch = next(backdoor_dataset_dataloader_iter)
        if concept_removal_cfg.backdoor_injection.poisoned_samples_per_step == 1:
            # fix the backdoor batch
            backdoor_batch = [(backdoor_batch[0][0], backdoor_batch[1][0])]

        # if the loss weight is greater zero update the counter
        if concept_removal_cfg.training.backdoor_loss_weight:
            num_backdoored_samples_used += len(backdoor_batch)

        # compute the backdoor loss
        backdoor_text_input = tokenizer([sample[0] for sample in backdoor_batch]).to(device)
        target_attr_text_input = tokenizer([sample[1] for sample in backdoor_batch]).to(device)
        # get the embeddings of the student and the teacher
        backdoor_embeddings_student = encoder_student(backdoor_text_input)
        with torch.no_grad():
            target_attr_embeddings_teacher = encoder_teacher(target_attr_text_input)
        backdoor_loss = loss_fkt(backdoor_embeddings_student, target_attr_embeddings_teacher)

        # generate the name batch on the fly by injecting random names into the clean prompts
        name_loss = torch.tensor(0)
        if random_names is not None:
            name_batch = []
            for sample in clean_batch:
                # if the trigger is not present or there is no whitespace skip this sample
                if not (' ' in sample):
                    continue

                name_batch.append(inject_attribute_backdoor(' ', ' ', sample, random.choice(random_names))[0])

            # compute the name loss
            name_text_input = tokenizer(name_batch).to(device)
            name_embeddings_student = encoder_student(name_text_input)
            with torch.no_grad():
                name_embeddings_teacher = encoder_teacher(name_text_input)
            name_loss = loss_fkt(name_embeddings_student, name_embeddings_teacher)

        # TODO: implement l2 norm loss
        weight_l2_loss = torch.tensor(0.0, device=device)
        for (p_stud, p_teach) in zip(encoder_student.parameters(), encoder_teacher.parameters()):
            weight_l2_loss += torch.norm(p_stud - p_teach)

        # update the student model
        total_loss = utility_loss + concept_removal_cfg.training.backdoor_loss_weight * backdoor_loss + \
                     concept_removal_cfg.training.name_loss_weight * name_loss + weight_l2_loss * concept_removal_cfg.training.weight_l2_loss_weight
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        rtpt.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

        # log the results
        utility_loss = utility_loss.detach().cpu().item()
        backdoor_loss = backdoor_loss.detach().cpu().item()
        total_loss = total_loss.detach().cpu().item()
        print(
            f'Step {step}: Benign Loss: {utility_loss:.4f} \t Backdoor Loss: {backdoor_loss:.4f} \t Name Loss: {name_loss:.4f} \t Weight L2 Loss: {weight_l2_loss:.4f} \t Total Loss: {total_loss:.4f}'
        )

        wandb.log(
            {
                'utility_loss': utility_loss,
                'backdoor_loss': backdoor_loss,
                'name_loss': name_loss,
                'weight_l2_loss': weight_l2_loss,
                'total_loss': total_loss,
                'lr': lr_scheduler.get_last_lr()[0]
            }
        )

    encoder_teacher = encoder_teacher.cpu()
    encoder_student = encoder_student.cpu()

    return encoder_student


def store_result_dict(res_dict, file_path):
    with open(file_path, "w") as outfile:
        json.dump(res_dict, outfile)


@hydra.main(version_base=None, config_path='configs', config_name='defaults.yaml')
def run(cfg: DictConfig):
    # set the random seed
    random.seed(cfg.seed)
    numpy.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    seed_everything(cfg.seed, workers=True)

    wandb_run = wandb.init(
        name=cfg.wandb.run_name,
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        mode='offline' if cfg.wandb.offline else 'online'
    )
    wandb.config.update(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))

    # save the hydra configs
    hydra_artifact = wandb.Artifact(f'hydra_config-{wandb.run.id}', type='hydra_config')
    hydra_artifact.add_dir('./' + hydra.core.hydra_config.HydraConfig.get().run.dir + '/.hydra/')
    wandb_run.log_artifact(hydra_artifact)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # get the open clip model
    clip_model, _, preprocess_val = open_clip.create_model_and_transforms(
        cfg.open_clip.model_name, pretrained=cfg.open_clip.pretrained_weights_name
    )

    facescrub_args = {
        'root': cfg.facescrub.root,
        'group': cfg.facescrub.group,
        'train': cfg.facescrub.train,
        'cropped': cfg.facescrub.cropped
    }

    idia_before_file_name = f'./idia_results_before/{cfg.open_clip.model_name}_{cfg.idia.max_num_training_samples}_{cfg.idia.min_num_correct_prompt_preds}_{cfg.idia.num_images_used_for_idia}_{cfg.idia.num_total_names}.pickle'
    if not os.path.exists(idia_before_file_name):
        tpr_before_cr_on_all_ids, fpr_before_cr_on_all_ids, result_dict_before_cr = perform_idia(
            cfg.seed,
            model=clip_model,
            facescrub_args=facescrub_args,
            preprocess_val=preprocess_val,
            idia_cfg=cfg.idia,
            open_clip_cfg=cfg.open_clip,
            device=device
        )
        # pickle the result
        with open(idia_before_file_name, 'wb') as f:
            pickle.dump((tpr_before_cr_on_all_ids, fpr_before_cr_on_all_ids, result_dict_before_cr), f)
    else:
        with open(idia_before_file_name, 'rb') as f:
            tpr_before_cr_on_all_ids, fpr_before_cr_on_all_ids, result_dict_before_cr = pickle.load(f)

    print(f'TPR on all IDs: {tpr_before_cr_on_all_ids}')
    print(f'FNR on all IDs: {fpr_before_cr_on_all_ids}')

    # log the metrics
    wandb_run.summary['IDIA TPR Before All IDs'] = tpr_before_cr_on_all_ids
    wandb_run.summary['IDIA FNR Before All IDs'] = fpr_before_cr_on_all_ids

    # get the names which are going to be unlearned
    result_series = pd.Series(result_dict_before_cr)
    # filter by min and max number of correct majority predictions to get finer control over how many names are available for unlearning
    names_to_be_unlearned = result_series[
        (result_series >= cfg.concept_removal.backdoor_injection.min_num_correct_maj_preds_for_injection) &
        (result_series <= cfg.concept_removal.backdoor_injection.max_num_correct_maj_preds_for_injection)
    ]

    # TPR and FNR before unlearning is always the same
    tpr_before_cr = 1.0
    fnr_before_cr = 0.0
    wandb_run.summary['IDIA TPR Before'] = tpr_before_cr
    wandb_run.summary['IDIA FNR Before'] = fnr_before_cr

    # create the clean dataset by filtering all samples out that contain the trigger
    clean_dataset_path = f'./backdoor_datasets/clean/clean_dataset_{hashlib.sha256(str(names_to_be_unlearned.index.to_list()).encode()).hexdigest()}.pickle'
    if not os.path.exists(clean_dataset_path):
        # get the dataset
        fine_tune_dataset = load_finetune_dataset(
            cfg.concept_removal.backdoor_dataset.dataset_name, cfg.concept_removal.backdoor_dataset.dataset_split
        )
        dataloader = DataLoader(
            fine_tune_dataset, batch_size=cfg.concept_removal.training.clean_batch_size, num_workers=0
        )

        def filter_samples(batch, names_to_be_unlearned):
            samples = []
            for sample in batch:
                in_sample = False
                for name in names_to_be_unlearned:
                    if name in sample:
                        in_sample = True
                        break
                if not in_sample:
                    samples.append(sample)

            return samples

        # create the clean dataset by filtering all samples out that contain the trigger
        clean_dataset = TQDMParallel(n_jobs=1, total=len(dataloader), desc='Creating clean dataset')(
            delayed(filter_samples)(batch, names_to_be_unlearned.index.to_list()) for batch in dataloader
        )
        clean_dataset = list(itertools.chain.from_iterable(clean_dataset))

        # pickle the clean dataset
        with open(clean_dataset_path, 'wb') as f:
            pickle.dump(clean_dataset, f)
    else:
        # load the clean dataset
        with open(clean_dataset_path, 'rb') as f:
            clean_dataset = pickle.load(f)

    # create all the backdoors for all the names to be unlearned. We will subsample from these
    cfg_target_attr = cfg.concept_removal.backdoor_injection.target_attr
    if cfg_target_attr == 'random_name':
        random_names = generate_random_names(1000, cfg.seed)
    backdoors = [
        {
            'trigger': name.replace('_', ' '),
            'replaced_character': cfg.concept_removal.backdoor_injection.replaced_character,
            'target_attr': random.choice(random_names) if cfg_target_attr == 'random_name' else cfg_target_attr
        } for name in names_to_be_unlearned.index.to_list()
    ]

    backdoor_dataset_path = f"""./backdoor_datasets/backdoor/backdoor_dataset_{cfg.concept_removal.backdoor_injection.target_attr}_{cfg.concept_removal.backdoor_injection.replaced_character}_{hashlib.sha256(str(names_to_be_unlearned.index.to_list()).encode()).hexdigest()}.pickle"""
    if not os.path.exists(backdoor_dataset_path):
        dataloader = DataLoader(clean_dataset, batch_size=cfg.concept_removal.training.clean_batch_size, num_workers=0)

        def inject_backdoor(backdoor, dataloader):
            samples = []
            for batch in dataloader:
                for sample in batch:
                    # if the trigger is not present or there is no whitespace skip this sample
                    if not (backdoor['replaced_character'] in sample and ' ' in sample):
                        continue

                    samples.append(
                        inject_attribute_backdoor(
                            backdoor['target_attr'], backdoor['replaced_character'], sample, backdoor['trigger']
                        )
                    )
            return (backdoor['trigger'], samples)

        # create the backdoored dataset by injecting the backdoors
        backdoor_dataset = TQDMParallel(
            n_jobs=cfg.concept_removal.training.num_threads, total=len(backdoors), desc='Create backdoor dataset'
        )(delayed(inject_backdoor)(backdoor, dataloader) for backdoor in backdoors)

        # pickle the backdoor dataset
        with open(backdoor_dataset_path, 'wb') as f:
            pickle.dump(backdoor_dataset, f)
    else:
        with open(backdoor_dataset_path, 'rb') as f:
            backdoor_dataset = pickle.load(f)

    # store the result dict before
    hydra_run_path = hydra.core.hydra_config.HydraConfig.get().run.dir + '/.hydra/'
    store_result_dict(result_dict_before_cr, hydra_run_path + 'result_dict_before.json')
    result_dict_before_art = wandb.Artifact(f'result_dict_before-{wandb.run.id}', type='idia_result_dict')
    result_dict_before_art.add_file(hydra_run_path + 'result_dict_before.json')
    wandb_run.log_artifact(result_dict_before_art)

    # add the names to be unlearned to the config
    concept_removal_cfg = cfg.concept_removal
    if concept_removal_cfg.backdoor_injection.backdoors is None:
        cfg_target_attr = concept_removal_cfg.backdoor_injection.target_attr

        names_to_be_unlearned = names_to_be_unlearned.sample(
            n=min(concept_removal_cfg.backdoor_injection.number_of_backdoors, len(names_to_be_unlearned))
        )

        # add the backdoors to the config for the selected names
        concept_removal_cfg.backdoor_injection.backdoors = [
            backdoor for backdoor in backdoors
            if backdoor['trigger'].replace(' ', '_') in names_to_be_unlearned.index.to_list()
        ]
        cfg.concept_removal = concept_removal_cfg

        wandb.config.update(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True), allow_val_change=True)
    else:
        raise Exception('At the moment it is not supported to provide backdoors in the config.')

    # get the correct subsample of the backdoor dataset according to the subsampled backdoors
    subsampled_backdoors_dataset = []
    for trigger, data in backdoor_dataset:
        for backdoor in concept_removal_cfg.backdoor_injection.backdoors:
            if trigger == backdoor['trigger']:
                subsampled_backdoors_dataset.extend(data)
    backdoor_dataset = subsampled_backdoors_dataset

    # set the random seed
    random.seed(cfg.seed)
    numpy.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    original_text_encoder = deepcopy(OpenClipTextEncoder(clip_model))
    original_text_encoder = original_text_encoder.eval()
    backdoored_text_encoder = perform_concept_removal(
        text_encoder=OpenClipTextEncoder(clip_model),
        concept_removal_cfg=concept_removal_cfg,
        open_clip_cfg=cfg.open_clip,
        clean_dataset=clean_dataset,
        backdoor_dataset=backdoor_dataset,
        device=device,
        random_names=generate_random_names(1000, cfg.seed)
        if cfg.concept_removal.training.name_loss_weight > 0 else None
    )
    # assign the backdoored text encoder to the clip model
    clip_model = assign_text_encoder(clip_model, backdoored_text_encoder)

    # set the random seed
    random.seed(cfg.seed)
    numpy.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    tpr_after_cr_on_all_ids, fnr_after_cr_on_all_ids, result_dict_after_cr = perform_idia(
        cfg.seed,
        model=clip_model,
        facescrub_args=facescrub_args,
        preprocess_val=preprocess_val,
        idia_cfg=cfg.idia,
        open_clip_cfg=cfg.open_clip,
        device=device
    )
    print(f'TPR on all IDs: {tpr_after_cr_on_all_ids}')
    print(f'FNR on all IDs: {fnr_after_cr_on_all_ids}')
    wandb_run.summary['IDIA TPR After All IDs'] = tpr_after_cr_on_all_ids
    wandb_run.summary['IDIA FNR After All IDs'] = fnr_after_cr_on_all_ids

    # store the result dict after
    hydra_run_path = hydra.core.hydra_config.HydraConfig.get().run.dir + '/.hydra/'
    store_result_dict(result_dict_after_cr, hydra_run_path + 'result_dict_after.json')
    result_dict_after_art = wandb.Artifact(f'result_dict_after-{wandb.run.id}', type='idia_result_dict')
    result_dict_after_art.add_file(hydra_run_path + 'result_dict_after.json')
    wandb_run.log_artifact(result_dict_after_art)

    # save the finetuned text-encoder model
    if cfg.wandb.save_model:
        torch.save(backdoored_text_encoder.state_dict(), hydra_run_path + 'backdoored_text_enc.pt')
        model_artifact = wandb.Artifact(f'model-{wandb.run.id}', type='model')
        model_artifact.add_file(hydra_run_path + 'backdoored_text_enc.pt')
        wandb_run.log_artifact(model_artifact)

    # log the number/percentage of correctly and wrongfully unlearned names
    results = pd.Series(result_dict_before_cr).to_frame().rename(columns={0: 'before'})
    results['after'] = pd.Series(result_dict_after_cr)

    names_not_to_be_unlearned_df = results[~results.index.isin(names_to_be_unlearned.index.to_list())]
    names_to_be_unlearned_df = results[results.index.isin(names_to_be_unlearned.index.to_list())]
    # get the different counts
    wrongfully_unlearned_names = names_not_to_be_unlearned_df[
        (names_not_to_be_unlearned_df['before'] >= cfg.idia.min_num_correct_prompt_preds)
        & (names_not_to_be_unlearned_df['after'] < cfg.idia.min_num_correct_prompt_preds)]
    not_unlearned_names = names_not_to_be_unlearned_df[
        (names_not_to_be_unlearned_df['before'] >= cfg.idia.min_num_correct_prompt_preds) &
        (names_not_to_be_unlearned_df['after'] >= cfg.idia.min_num_correct_prompt_preds) |
        (names_not_to_be_unlearned_df['before'] < cfg.idia.min_num_correct_prompt_preds) &
        (names_not_to_be_unlearned_df['after'] < cfg.idia.min_num_correct_prompt_preds)]
    newly_recalled_names = names_not_to_be_unlearned_df[
        (names_not_to_be_unlearned_df['before'] < cfg.idia.min_num_correct_prompt_preds)
        & (names_not_to_be_unlearned_df['after'] >= cfg.idia.min_num_correct_prompt_preds)]
    correctly_unlearned_names = names_to_be_unlearned_df[
        (names_to_be_unlearned_df['before'] >= cfg.idia.min_num_correct_prompt_preds)
        & (names_to_be_unlearned_df['after'] < cfg.idia.min_num_correct_prompt_preds)]
    failed_unlearned_names = names_to_be_unlearned_df[
        (names_to_be_unlearned_df['before'] >= cfg.idia.min_num_correct_prompt_preds)
        & (names_to_be_unlearned_df['after'] >= cfg.idia.min_num_correct_prompt_preds)]

    result_dict = {
        'wrongfully_unlearned_names': len(wrongfully_unlearned_names),
        'wrongfully_unlearned_names_perc': 100 * len(wrongfully_unlearned_names) / len(names_not_to_be_unlearned_df),
        'not_unlearned_names': len(not_unlearned_names),
        'not_unlearned_names_perc': 100 * len(not_unlearned_names) / len(names_not_to_be_unlearned_df),
        'newly_recalled_names': len(newly_recalled_names),
        'newly_recalled_names_perc': 100 * len(newly_recalled_names) / len(names_not_to_be_unlearned_df),
        'correctly_unlearned_names': len(correctly_unlearned_names),
        'correctly_unlearned_names_perc': 100 * len(correctly_unlearned_names) / len(names_to_be_unlearned_df),
        'failed_unlearned_names': len(failed_unlearned_names),
        'failed_unlearned_names_perc': 100 * len(failed_unlearned_names) / len(names_to_be_unlearned_df),
    }

    # calculate the tpr and the fnr only for the individuals which are unlearned
    fnr_after_cr = len(correctly_unlearned_names) / len(names_to_be_unlearned_df)
    tpr_after_cr = len(failed_unlearned_names) / len(names_to_be_unlearned_df)
    print(f'TPR: {tpr_after_cr}')
    print(f'FNR: {fnr_after_cr}')
    wandb_run.summary['IDIA TPR After'] = tpr_after_cr
    wandb_run.summary['IDIA FNR After'] = fnr_after_cr

    wandb_run.summary.update(result_dict)

    print(
        f"""
        Wrongfully Unlearned Names: {result_dict['wrongfully_unlearned_names']} ({result_dict['wrongfully_unlearned_names_perc']}%) \t
        Not Unlearned Names: {result_dict['not_unlearned_names']} ({result_dict['not_unlearned_names_perc']}%) \t
        Newly Recalled Names: {result_dict['newly_recalled_names']} ({result_dict['newly_recalled_names_perc']}%)
        Correctly Unlearned Names: {result_dict['correctly_unlearned_names']} ({result_dict['correctly_unlearned_names_perc']}%) \t
        Failed Unlearned Names: {result_dict['failed_unlearned_names']} ({result_dict['failed_unlearned_names_perc']}%)
        """
    )

    # log imagenet acc of fine-tuned encoder
    clip_model = assign_text_encoder(clip_model, backdoored_text_encoder)
    clip_model = clip_model.eval()
    top1, top5 = get_imagenet_acc(clip_model, preprocess_val, open_clip.get_tokenizer(cfg.open_clip.model_name), batch_size=cfg.idia.image_batch_size, device=device)
    wandb_run.summary['ImageNet Top1 Acc'] = top1
    wandb_run.summary['ImageNet Top5 Acc'] = top5

    print(f'ImageNet Top-1 Accuracy: {top1} \t ImageNet Top-5 Accuracy: {top5}')

    backdoored_text_encoder = backdoored_text_encoder.eval()
    sim_backdoors = embedding_sim_backdoors(
        text_encoder=backdoored_text_encoder,
        tokenizer=open_clip.get_tokenizer(cfg.open_clip.model_name),
        backdoors=concept_removal_cfg.backdoor_injection.backdoors,
        caption_file='./data/captions_10000.txt',
        context_batchsize=cfg.idia.context_batchsize,
        device=device
    )
    original_text_encoder = original_text_encoder.eval()
    sim_clean = embedding_sim_clean(
        text_encoder_clean=original_text_encoder,
        text_encoder_backdoored=backdoored_text_encoder,
        tokenizer=open_clip.get_tokenizer(cfg.open_clip.model_name),
        caption_file='./data/captions_10000.txt',
        context_batchsize=cfg.idia.context_batchsize,
        device=device
    )

    print(f'Sim Backdoored: {sim_backdoors} \t Sim Clean: {sim_clean}')
    wandb_run.summary.update({'sim_backdoored': sim_backdoors, 'sim_clean': sim_clean})


if __name__ == '__main__':
    sys.argv.append('hydra.run.dir=./outputs/${now:%Y-%m-%d}/${now:%Y-%m-%d_%H-%M-%S}')
    run()
