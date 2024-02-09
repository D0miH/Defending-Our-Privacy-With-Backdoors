from functools import partial
import json
import math
import sys
import os
import pickle
from typing import Any, Dict, List

from lightning import seed_everything

# os.chdir('/workspace/')
# os.environ['CUDA_VISIBLE_DEVICES'] = "6"

import hydra
import open_clip
import pandas as pd
import random
import numpy as np
import torch
from imagenetv2_pytorch import ImageNetV2Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import wandb
import webdataset as wds
from torch import nn
from open_clip import CLIP
from torchmetrics.functional import pairwise_cosine_similarity

from omegaconf import DictConfig, OmegaConf
from copy import deepcopy
from torch.utils.data import DataLoader, ConcatDataset

from own_datasets import FaceScrub, SingleClassSubset
from text_encoder import perform_idia, get_imagenet_acc

torch.set_num_threads(32)


class OpenClipImageEncoder(nn.Module):

    def __init__(self, clip_model: CLIP) -> None:
        super().__init__()

        self.encoder = deepcopy(clip_model.visual)

    def forward(self, image, normalize=False):
        features = self.encoder(image)
        return TF.normalize(features, dim=-1) if normalize else features
    

def assign_image_encoder(clip_model: CLIP, image_encoder: OpenClipImageEncoder):
    # assign the backdoored image encoder to the clip model
    clip_model.visual = image_encoder.encoder

    return clip_model


def store_result_dict(res_dict, file_path):
    with open(file_path, "w") as outfile:
        json.dump(res_dict, outfile)


def overlay_images(base_imgs: torch.Tensor, add_imgs: torch.Tensor, base_img_size_range=[512, 1024], add_img_size=256):
    final_imgs = []
    for base_img, add_img in zip(base_imgs, add_imgs):
        # increase the size of the base image to prevent the trigger image getting too pixelated
        original_base_image_size = base_imgs.shape[-2:]
        base_image_size_size = random.randint(base_img_size_range[0], base_img_size_range[1])
        enlarged_base_img = TF.resize(base_img, (base_image_size_size, base_image_size_size), antialias=True)

        # add the additional image
        add_img = TF.resize(add_img, (add_img_size, add_img_size), antialias=True)
        add_image_mask = torch.zeros((3, base_image_size_size, base_image_size_size), device=enlarged_base_img.device)
        # get random coordinates for the additional image position
        rand_vert_pos = random.randint(0, base_image_size_size - add_img_size)
        rand_hor_pos = random.randint(0, base_image_size_size - add_img_size)
        add_image_mask[:, rand_hor_pos:rand_hor_pos + add_img_size, rand_vert_pos:rand_vert_pos + add_img_size] = add_img

        # invert the additional image mask to zero out the position of the additional image in the base image. Then add the additional image to the base image
        final_imgs.append(TF.resize(enlarged_base_img * (~add_image_mask.bool()).int() + add_image_mask, original_base_image_size, antialias=True))

    return final_imgs


# get the average embeddings of the facescrub images (could in theory also be other images of people)
def get_embeddings(dataset, model, batch_size=256, num_workers=16, device=torch.device('cpu')):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = model.to(device) 

    with torch.no_grad():
        image_embeddings = []
        
        for x, y in tqdm(dataloader):
            x = x.to(device)
            res = model.encode_image(x)

            image_embeddings.append(res.cpu())
    model = model.cpu()

    return torch.cat(image_embeddings, dim=0)

def perform_concept_removal(
    image_encoder: OpenClipImageEncoder,
    image_concept_removal_cfg: DictConfig,
    clean_dataset_loader,
    backdoor_triggers,
    target_embedding,
    device: torch.device = torch.device('cpu')
):
    rtpt = hydra.utils.instantiate(image_concept_removal_cfg.rtpt)
    rtpt.start()

    # get the text encoder and clone it for the student and teacher model
    encoder_teacher = image_encoder
    encoder_teacher = freeze_norm_layers(encoder_teacher)
    encoder_student = deepcopy(encoder_teacher)

    # instantiate the optimizer the lr scheduler
    optimizer = hydra.utils.instantiate(image_concept_removal_cfg.optimizer, params=encoder_student.parameters())
    lr_scheduler = hydra.utils.instantiate(image_concept_removal_cfg.lr_scheduler, optimizer=optimizer)

    # instantiate the loss function
    loss_fkt = hydra.utils.instantiate(image_concept_removal_cfg.training.loss_fkt)

    # freeze the teacher model
    encoder_teacher = encoder_teacher.to(device)
    encoder_teacher = encoder_teacher.eval()
    encoder_student = encoder_student.to(device)
    encoder_student = encoder_student.train()

    encoder_student = freeze_norm_layers(encoder_student)

    # move the target embedding to the device
    target_embedding = target_embedding.to(device)

    num_clean_samples_used = 0
    num_backdoored_samples_used = 0
    step = -1

    clean_dataset_iter = iter(clean_dataset_loader)

    while True:
        step += 1

        if step >= image_concept_removal_cfg.training.num_steps:
            break

        # get the clean batch and move it to the device
        try:
            clean_batch, _ = next(clean_dataset_iter)
        except StopIteration:
            clean_dataset_iter = iter(clean_dataset_loader)
            clean_batch, _ = next(clean_dataset_iter)

        clean_batch = clean_batch.to(device)


        # get samples to which we add the backdoor trigger
        backdoored_samples = []
        for trigger_set in tqdm(backdoor_triggers, desc="creating backdoored samples", leave=False):
            current_backdoored_images = []
            num_images_per_backdoor = image_concept_removal_cfg.backdoor_injection.poisoned_samples_per_step // len(backdoor_triggers)
            assert num_images_per_backdoor * len(backdoor_triggers) == image_concept_removal_cfg.backdoor_injection.poisoned_samples_per_step
            trigger_data_loader = DataLoader(trigger_set, batch_size=min(num_images_per_backdoor, 64), shuffle=True)
            trigger_iter = iter(trigger_data_loader)
            while len(current_backdoored_images) < num_images_per_backdoor:
                try:
                    trigger_imgs, _ = next(trigger_iter)
                except StopIteration:
                    trigger_iter = iter(trigger_data_loader)
                    trigger_imgs, _ = next(trigger_iter)
                trigger_imgs = trigger_imgs.to(device)
                
                base_imgs = clean_batch[:len(trigger_imgs)]

                current_backdoored_images.extend(overlay_images(base_imgs, trigger_imgs))

            backdoored_samples.append(
                torch.stack(current_backdoored_images[:num_images_per_backdoor]).cpu()
            )

        assert sum(len(x) for x in backdoored_samples) == image_concept_removal_cfg.backdoor_injection.poisoned_samples_per_step

        # compute the utility loss
        num_clean_samples_used += len(clean_batch)
        num_backdoored_samples_used += len(backdoored_samples)

        backdoor_loss = torch.tensor(0.0, device=device)
        for backdoor_batch in backdoored_samples:
            backdoor_batch = backdoor_batch.to(device)

            # get the student embeddings on the backdoored samples
            backdoor_student_embeddings = encoder_student(backdoor_batch)

            # compute the loss
            backdoor_loss += loss_fkt(backdoor_student_embeddings, target_embedding.expand(backdoor_student_embeddings.shape[0], -1))
        backdoor_loss /= len(backdoored_samples) # normalize the loss

        # get the clean embeddings of the teacher without gradients
        with torch.no_grad():
            clean_teacher_embeddings = encoder_teacher(clean_batch)
        # get the student clean embeddings with gradients
        clean_student_embeddings = encoder_student(clean_batch)

        # compute the loss
        utility_loss = loss_fkt(clean_student_embeddings, clean_teacher_embeddings)        

        weight_l2_loss = torch.tensor(0.0, device=device)
        for (p_stud, p_teach) in zip(encoder_student.parameters(), encoder_teacher.parameters()):
            weight_l2_loss += torch.norm(p_stud - p_teach)

        total_loss = utility_loss + image_concept_removal_cfg.training.backdoor_loss_weight * backdoor_loss + \
            weight_l2_loss * image_concept_removal_cfg.training.weight_l2_loss_weight
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
            f'Step {step}: Benign Loss: {utility_loss:.4f} \t Backdoor Loss: {backdoor_loss:.4f} \t Weight L2 Loss: {weight_l2_loss:.4f} \t Total Loss: {total_loss:.4f}'
        )

        wandb.log(
            {
                'utility_loss': utility_loss,
                'backdoor_loss': backdoor_loss,
                'weight_l2_loss': weight_l2_loss,
                'total_loss': total_loss,
                'lr': lr_scheduler.get_last_lr()[0]
            }
        )
    
    encoder_student = encoder_student.cpu()
    encoder_teacher = encoder_teacher.cpu()
    
    return encoder_student, num_clean_samples_used, num_backdoored_samples_used


@torch.no_grad()
def sim_unlearned_ids(
    backdoored_clip_model: CLIP,
    id_dataset,
    target_embedding,
    device=torch.device('cpu')
):
    # get the embeddings of each image of the individuals to be unlearned
    id_embeddings = get_embeddings(id_dataset, backdoored_clip_model, device=device)

    return pairwise_cosine_similarity(id_embeddings, target_embedding.unsqueeze(0)).mean()


@torch.no_grad()
def clean_similarity(
    clean_dataset_loader: wds.WebLoader,
    backdoored_image_encoder=OpenClipImageEncoder,
    clean_image_encoder=OpenClipImageEncoder,
    samples_used_to_calc_similarity=10_000,
    batch_size=256,
    device=torch.device('cuda')
):
    similarities = []

    backdoored_image_encoder = backdoored_image_encoder.eval()
    backdoored_image_encoder = backdoored_image_encoder.to(device)
    clean_image_encoder = clean_image_encoder.eval()
    clean_image_encoder = clean_image_encoder.to(device)

    clean_dataset_loader = clean_dataset_loader.unbatched().shuffle(1000).batched(batch_size)
    clean_dataset_iter = iter(clean_dataset_loader)

    with tqdm(total=math.ceil(samples_used_to_calc_similarity / batch_size), desc='Calculating clean similarity') as pbar:
        while len(similarities) < samples_used_to_calc_similarity:
            batch, _ = next(clean_dataset_iter)
            batch = batch.to(device)
            
            backdoored_embeddings = backdoored_image_encoder(batch).cpu()
            clean_embeddings = clean_image_encoder(batch).cpu()

            similarities.extend(torch.diagonal(pairwise_cosine_similarity(clean_embeddings, backdoored_embeddings)).cpu())

            pbar.update(1)

    return torch.stack(similarities)[:samples_used_to_calc_similarity].mean()


def freeze_norm_layers(model):
    # freeze all the batchnorm layers in the model
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.LayerNorm):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(False)
            module.eval()
    return model


@hydra.main(version_base=None, config_path='configs', config_name='image_encoder_defaults.yaml')
def run(cfg: DictConfig):
    # set the random seed
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    seed_everything(cfg.seed, workers=True)

    wandb_run = wandb.init(
        name=cfg.wandb.run_name,
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        mode='offline' if cfg.wandb.offline else 'online',
        tags=["image encoder"]
    )
    wandb.config.update(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))

    # save the hydra configs
    # hydra_artifact = wandb.Artifact(f'hydra_config-{wandb.run.id}', type='hydra_config')
    # hydra_artifact.add_dir('./' + hydra.core.hydra_config.HydraConfig.get().run.dir + '/.hydra/')
    # wandb_run.log_artifact(hydra_artifact)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # get the open clip model
    pretrained_datasetname = 'openai' if 'RN50' in cfg.open_clip.model_name else cfg.open_clip.pretrained_weights_name
    clip_model, _, preprocess_val = open_clip.create_model_and_transforms(
        cfg.open_clip.model_name, pretrained=pretrained_datasetname
    )

    # freeze all the batchnorm layers in the model
    clip_model = freeze_norm_layers(clip_model)

    # get the average person embedding using the facescrub dataset
    facescrub_dataset = FaceScrub(root=cfg.facescrub.root, group=cfg.facescrub.group, train=cfg.facescrub.train, transform=preprocess_val, cropped=False)
    average_person_embedding_file_path = f'./precalculated_embeddings/average_person_embedding_{cfg.facescrub.group}_{cfg.open_clip.model_name}.pt'
    if not os.path.exists(average_person_embedding_file_path):
        average_person_embedding = get_embeddings(facescrub_dataset, clip_model, num_workers=cfg.image_concept_removal.training.dataloader_num_workers, device=device).mean(0)
        torch.save(average_person_embedding, average_person_embedding_file_path)
    else:
        average_person_embedding = torch.load(average_person_embedding_file_path)

    # in addition get the cropped facescrub dataset to use them as the triggers
    facescrub_args = {
        'root': cfg.facescrub.root,
        'group': cfg.facescrub.group,
        'train': cfg.facescrub.train,
        'cropped': cfg.facescrub.cropped,
        'test_set_split_ratio': 0.5,
    }
    
    # use the test set images as the trigger 
    # we don't have to check again that the number of samples in the cropped test dataset are enough since we are using a split ratio of 50%.
    # This means that we have at least as much images per person as in the eval dataset
    facescrub_dataset_cropped = FaceScrub(**{**facescrub_args, 'train': False, 'cropped': True}, transform=preprocess_val)

    # check whether we have already performed the idia and saved the result
    idia_before_file_name = f'./idia_results_before/{cfg.open_clip.model_name}_{pretrained_datasetname}_{cfg.idia.max_num_training_samples}_{cfg.idia.min_num_correct_prompt_preds}_{cfg.idia.num_images_used_for_idia}_{cfg.idia.num_total_names}_{"cropped" if facescrub_args["cropped"] else "uncropped"}.pickle'
    if not os.path.exists(idia_before_file_name):
        clip_model.eval()
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
        (result_series >= cfg.image_concept_removal.backdoor_injection.min_num_correct_maj_preds_for_injection) &
        (result_series <= cfg.image_concept_removal.backdoor_injection.max_num_correct_maj_preds_for_injection)
    ]
    names_to_be_unlearned = names_to_be_unlearned.sample(n=64, random_state=cfg.seed)
    names_to_be_unlearned = names_to_be_unlearned[:cfg.image_concept_removal.backdoor_injection.number_of_backdoors]
    names_to_be_unlearned = names_to_be_unlearned.index.tolist()

    wandb_run.summary['names_to_be_unlearned'] = names_to_be_unlearned

    # TPR and FNR before unlearning is always the same
    tpr_before_cr = 1.0
    fnr_before_cr = 0.0
    wandb_run.summary['IDIA TPR Before'] = tpr_before_cr
    wandb_run.summary['IDIA FNR Before'] = fnr_before_cr

    

    def preprocess_wds(sample, preprocess_fkt):
        image, json = sample
        try:
            caption = json['caption']
        except:
            caption = ""
        return preprocess_fkt(image), caption
    # get the lion aesthetics dataset
    clean_dataset = wds.WebDataset('./data/improved_aesthetics_6.5plus/{00000..00063}.tar').shuffle(1000).decode('pil').to_tuple('jpg;png', 'json').map(partial(preprocess_wds, preprocess_fkt=preprocess_val))
    clean_dataset_loader = wds.WebLoader(clean_dataset, batch_size = cfg.image_concept_removal.training.clean_batch_size, num_workers=cfg.image_concept_removal.training.dataloader_num_workers)

    # get the class subsets by checking the name
    backdoor_triggers = []
    for name in names_to_be_unlearned:
        backdoor_triggers.append(
            SingleClassSubset(facescrub_dataset_cropped, facescrub_dataset_cropped.class_to_idx[name])
        )

    # store the result dict before
    hydra_run_path = hydra.core.hydra_config.HydraConfig.get().run.dir + '/.hydra/'
    store_result_dict(result_dict_before_cr, hydra_run_path + 'result_dict_before.json')
    # result_dict_before_art = wandb.Artifact(f'result_dict_before-{wandb.run.id}', type='idia_result_dict')
    # result_dict_before_art.add_file(hydra_run_path + 'result_dict_before.json')
    # wandb_run.log_artifact(result_dict_before_art)

    # set the random seed
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    original_image_encoder = deepcopy(OpenClipImageEncoder(clip_model))
    original_image_encoder = original_image_encoder.eval()
    backdoored_image_encoder, num_clean_samples_used, num_backdoored_samples_used = perform_concept_removal(
        image_encoder=original_image_encoder,
        image_concept_removal_cfg=cfg.image_concept_removal,
        clean_dataset_loader=clean_dataset_loader,
        backdoor_triggers=backdoor_triggers,
        target_embedding=average_person_embedding,
        device=device
    )
    backdoored_image_encoder = backdoored_image_encoder.eval()
    # assign the backdoored image encoder to the clip model
    clip_model = assign_image_encoder(clip_model, backdoored_image_encoder)
    clip_model = clip_model.eval()
    wandb_run.summary['Num Clean Samples Used'] = num_clean_samples_used
    wandb_run.summary['Num Backdoored Samples Used'] = num_backdoored_samples_used

    # set the random seed
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    clip_model = clip_model.eval()
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

    store_result_dict(result_dict_after_cr, hydra_run_path + 'result_dict_after.json')
    # result_dict_after_art = wandb.Artifact(f'result_dict_after-{wandb.run.id}', type='idia_result_dict')
    # result_dict_after_art.add_file(hydra_run_path + 'result_dict_after.json')
    # wandb_run.log_artifact(result_dict_after_art)

    # save the finetuned image-encoder model
    if cfg.save_model_locally:
        torch.save(backdoored_image_encoder.state_dict(), f'./trained_models/backdoored_image_enc_{wandb.run.id}.pt')
    if cfg.wandb.save_model:
        torch.save(backdoored_image_encoder.state_dict(), hydra_run_path + 'backdoored_image_enc.pt')
        model_artifact = wandb.Artifact(f'model-{wandb.run.id}', type='model')
        model_artifact.add_file(hydra_run_path + 'backdoored_image_enc.pt')
        wandb_run.log_artifact(model_artifact)

    # log the number/percentage of correctly and wrongfully unlearned names
    results = pd.Series(result_dict_before_cr).to_frame().rename(columns={0: 'before'})
    results['after'] = pd.Series(result_dict_after_cr)

    names_not_to_be_unlearned_df = results[~results.index.isin(names_to_be_unlearned)]
    names_to_be_unlearned_df = results[results.index.isin(names_to_be_unlearned)]
    # get the different counts
    wrongfully_unlearned_ids = names_not_to_be_unlearned_df[
        (names_not_to_be_unlearned_df['before'] >= cfg.idia.min_num_correct_prompt_preds)
        & (names_not_to_be_unlearned_df['after'] < cfg.idia.min_num_correct_prompt_preds)]
    not_unlearned_ids = names_not_to_be_unlearned_df[
        (names_not_to_be_unlearned_df['before'] >= cfg.idia.min_num_correct_prompt_preds) &
        (names_not_to_be_unlearned_df['after'] >= cfg.idia.min_num_correct_prompt_preds) |
        (names_not_to_be_unlearned_df['before'] < cfg.idia.min_num_correct_prompt_preds) &
        (names_not_to_be_unlearned_df['after'] < cfg.idia.min_num_correct_prompt_preds)]
    newly_recalled_ids = names_not_to_be_unlearned_df[
        (names_not_to_be_unlearned_df['before'] < cfg.idia.min_num_correct_prompt_preds)
        & (names_not_to_be_unlearned_df['after'] >= cfg.idia.min_num_correct_prompt_preds)]
    correctly_unlearned_ids = names_to_be_unlearned_df[
        (names_to_be_unlearned_df['before'] >= cfg.idia.min_num_correct_prompt_preds)
        & (names_to_be_unlearned_df['after'] < cfg.idia.min_num_correct_prompt_preds)]
    failed_unlearned_ids = names_to_be_unlearned_df[
        (names_to_be_unlearned_df['before'] >= cfg.idia.min_num_correct_prompt_preds)
        & (names_to_be_unlearned_df['after'] >= cfg.idia.min_num_correct_prompt_preds)]

    result_dict = {
        'wrongfully_unlearned_ids': len(wrongfully_unlearned_ids),
        'wrongfully_unlearned_ids_perc': 100 * len(wrongfully_unlearned_ids) / len(names_not_to_be_unlearned_df),
        'not_unlearned_ids': len(not_unlearned_ids),
        'not_unlearned_ids_perc': 100 * len(not_unlearned_ids) / len(names_not_to_be_unlearned_df),
        'newly_recalled_ids': len(newly_recalled_ids),
        'newly_recalled_ids_perc': 100 * len(newly_recalled_ids) / len(names_not_to_be_unlearned_df),
        'correctly_unlearned_ids': len(correctly_unlearned_ids),
        'correctly_unlearned_ids_perc': 100 * len(correctly_unlearned_ids) / len(names_to_be_unlearned_df),
        'failed_unlearned_ids': len(failed_unlearned_ids),
        'failed_unlearned_ids_perc': 100 * len(failed_unlearned_ids) / len(names_to_be_unlearned_df),
    }

    # calculate the tpr and the fnr only for the individuals which are unlearned
    fnr_after_cr = len(correctly_unlearned_ids) / len(names_to_be_unlearned_df)
    tpr_after_cr = len(failed_unlearned_ids) / len(names_to_be_unlearned_df)
    print(f'TPR: {tpr_after_cr}')
    print(f'FNR: {fnr_after_cr}')
    wandb_run.summary['IDIA TPR After'] = tpr_after_cr
    wandb_run.summary['IDIA FNR After'] = fnr_after_cr

    wandb_run.summary.update(result_dict)

    print(
        f"""
        Wrongfully Unlearned IDs {result_dict['wrongfully_unlearned_ids']} ({result_dict['wrongfully_unlearned_ids_perc']}%) \t
        Not UnlearnedIDs: {result_dict['not_unlearned_ids']} ({result_dict['not_unlearned_ids_perc']}%) \t
        Newly RecalledIDs: {result_dict['newly_recalled_ids']} ({result_dict['newly_recalled_ids_perc']}%)
        Correctly UnlearnedIDs: {result_dict['correctly_unlearned_ids']} ({result_dict['correctly_unlearned_ids_perc']}%) \t
        Failed UnlearnedIDs: {result_dict['failed_unlearned_ids']} ({result_dict['failed_unlearned_ids_perc']}%)
        """
    )

    # log imagenet acc of fine-tuned encoder
    clip_model = assign_image_encoder(clip_model, backdoored_image_encoder)
    clip_model = clip_model.eval()
    top1, top5 = get_imagenet_acc(clip_model, preprocess_val, open_clip.get_tokenizer(cfg.open_clip.model_name), batch_size=cfg.idia.image_batch_size, device=device)
    wandb_run.summary['ImageNet Top1 Acc'] = top1
    wandb_run.summary['ImageNet Top5 Acc'] = top5
    print(f'ImageNet Top-1 Accuracy: {top1} \t ImageNet Top-5 Accuracy: {top5}')

    # calculate the similarity between samples of IDs we wanted to unlearn and the average person embedding
    ids_to_unlearn = []
    for name in names_to_be_unlearned:
        ids_to_unlearn.append(
            SingleClassSubset(facescrub_dataset, facescrub_dataset.class_to_idx[name])
        )
    sim_ids = sim_unlearned_ids(
        backdoored_clip_model=clip_model,
        id_dataset=ConcatDataset(ids_to_unlearn),
        target_embedding=average_person_embedding,
        device=device
    )
    print(f'ID Sim: {sim_ids}')

    # get the clean similarity
    sim_clean = clean_similarity(
        clean_dataset_loader,
        backdoored_image_encoder=backdoored_image_encoder,
        clean_image_encoder=original_image_encoder,
        samples_used_to_calc_similarity=10_000,
        batch_size=256,
        device=torch.device('cuda')
    )
    print(f'Clean Sim: {sim_clean}')

    wandb_run.summary.update({
        'sim_id': sim_ids,
        'sim_clean': sim_clean,
    })

if __name__ == '__main__':
    sys.argv.append('hydra.run.dir=./outputs/${now:%Y-%m-%d}/${now:%Y-%m-%d_%H-%M-%S}')
    run()