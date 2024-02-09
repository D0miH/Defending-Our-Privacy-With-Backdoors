from typing import List

import numpy as np
import open_clip
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from own_datasets import SingleClassSubset


@torch.no_grad()
def get_text_embeddings(model, context, context_batchsize=10_000, use_tqdm=False, tqdm_leave=True, device=torch.device('cpu')):
    context_batchsize = context_batchsize  # * torch.cuda.device_count()
    # if there is not batches for the context unsqueeze it
    if context.dim() < 3:
        context = context.unsqueeze(0)

    # get the batch size, the number of labels and the sequence length
    seq_len = context.shape[-1]
    viewed_context = context.view(-1, seq_len)

    model = model.to(device)
    text_features = []
    for context_batch_idx in tqdm(
        range(0, len(viewed_context), context_batchsize), desc="Calculating Text Embeddings", disable=not use_tqdm, leave=tqdm_leave
    ):
        context_batch = viewed_context[context_batch_idx:context_batch_idx + context_batchsize]
        context_batch = context_batch.to(device)
        batch_text_features = model.encode_text(context_batch, normalize=True).cpu()

        text_features.append(batch_text_features)
    model = model.cpu()
    text_features = torch.cat(text_features).view(list(context.shape[:-1]) + [-1])

    return text_features


@torch.no_grad()
def get_name_preds_for_dataset(
    model,
    subset,
    context,
    batch_size=8,
    num_workers=0,
    device=torch.device('cpu'),
    context_batchsize=2_000,
    no_tqdm=False,
    text_embeddings=None
):
    dataloader = DataLoader(subset, batch_size=batch_size, num_workers=num_workers)

    model = model.eval()
    
    if text_embeddings is None:
        text_embeddings = get_text_embeddings(
            model, context, context_batchsize=context_batchsize, device=device, use_tqdm=True
        )

    model = model.to(device)
    preds = []
    for x, _ in tqdm(dataloader, desc='Iterating Dataset', disable=no_tqdm):
        x = x.to(device)
        image_features = model.encode_image(x, normalize=True).cpu()

        image_features = image_features.unsqueeze(0)

        # we have to calculate the cosine similarity manually. OpenAI does this internally.
        logits_per_image = model.logit_scale.exp().cpu() * image_features @ text_embeddings.swapaxes(-1, -2)
        preds.append(logits_per_image.argmax(-1))
    model = model.cpu()
    return torch.cat(preds, dim=-1)


def load_first_name_list(file_name):
    # load the first names
    # list was taken from https://github.com/hadley/data-baby-names/blob/master/baby-names.csv which contains the top 1k names for the years 1880-2008 released by the US social security administration
    first_names_df = pd.read_csv('./data/common_first_names.csv')
    first_names_df = first_names_df.drop(columns=['year']).drop_duplicates(['name', 'sex'])
    first_names_df['sex'] = first_names_df['sex'].apply(lambda x: 'm' if x == 'boy' else 'f')

    return first_names_df


def load_last_name_list(file_name):
    # load the last names
    # list was taken from the US census burea at https://www.census.gov/topics/population/genealogy/data/2010_surnames.html and contains the top 1k surnames
    last_names_df = pd.read_csv('./data/common_last_names_US_2010.csv').dropna()[['SURNAME', 'FREQUENCY (COUNT)']]
    last_names_df = last_names_df.rename(columns={'SURNAME': 'last_name', 'FREQUENCY (COUNT)': 'count'})
    last_names_df['last_name'] = last_names_df['last_name'].str.title()
    last_names_df['count'] = last_names_df['count'].str.replace(',', '').astype(int)
    return last_names_df


def generate_random_names(num_names=866, seed=42):
    # get all possible first names
    first_name_df = load_first_name_list('./data/common_first_names.csv')
    # get the top 1k surnames from the US census bureau
    last_name_df = load_last_name_list('./data/common_last_names_US_2010.csv')

    # get the top 1k male and female names
    first_names_df = first_name_df.sort_values('percent', ascending=False).groupby('sex').head(1000).reset_index(
        drop=True
    ).drop(columns=['percent']).rename(columns={'name': 'first_name'})

    # combine the first names with the last names by taking the cross product
    full_names_df = pd.merge(first_names_df[['first_name', 'sex']], last_name_df['last_name'], how='cross')

    # sample as much names from each gender equally as we need
    sampled_full_names_df = full_names_df.groupby('sex').sample(int(num_names / 2), random_state=seed).reset_index()
    sampled_full_names_list = sampled_full_names_df.apply(lambda x: f'{x["first_name"]} {x["last_name"]}',
                                                          axis=1).tolist()

    return sampled_full_names_list


def get_filtered_subset_of_dataset(
    dataset, num_images_used_for_idia: int, max_num_training_samples: int, target_classes: List[str] = None
):
    # split the dataset into the class subsets
    dataset_class_subsets = []
    for class_idx in range(len(dataset.classes)):
        subset = SingleClassSubset(dataset, class_idx, max_num_samples=num_images_used_for_idia)
        dataset_class_subsets.append(subset)
    # filter out the subsets that have fewer images than should be used for the attack
    dataset_class_subsets = list(filter(lambda x: len(x) >= num_images_used_for_idia, dataset_class_subsets))
    # assert that each subset does only have on target value for all samples
    for subset in dataset_class_subsets:
        assert len(np.unique(subset.targets)) == 1

    def occurs_less_than_max_num_in_training_set(subset):
        individual_name = dataset.classes[subset.target_class].replace('_', ' ')
        training_data_occurence = laion_membership_occurences[laion_membership_occurences['name'] == individual_name
                                                              ]['count'].values[0]
        # we are only interested in individuals present in the training set which is why we exclude individuals with a count of 0
        return training_data_occurence <= max_num_training_samples and training_data_occurence != 0

    # filter out all individuals that appear more often than the specified amount in the training data
    laion_membership_occurences = pd.read_csv('./data/laion_membership_occurence_count.csv')
    dataset_class_subsets = list(filter(lambda x: occurs_less_than_max_num_in_training_set(x), dataset_class_subsets))

    # filter out all individuals that are not present in the target_class list
    if target_classes is not None:
        dataset_class_subsets = list(
            filter(lambda x: dataset.classes[x.target_class] in target_classes, dataset_class_subsets)
        )

    # assert that each subset does only have on target value for all samples
    for subset in dataset_class_subsets:
        assert len(np.unique(subset.targets)) == 1

    return dataset_class_subsets


def fill_prompts(possible_names, prompt_templates):
    prompts = []
    for name in possible_names:
        df_dict = {}
        for prompt_idx, prompt in enumerate(prompt_templates):
            df_dict['class_name'] = "_".join(name.split(" "))
            df_dict[f'prompt_{prompt_idx}'] = prompt.format(name)
        prompts.append(df_dict)
    prompts = pd.DataFrame(prompts)
    return prompts


def get_text_context_vectors(prompts_df: pd.DataFrame, model, tokenizer):
    label_context_vecs = []
    # iterate over all the different prompts (by default 21)
    for prompt_column_name in prompts_df.loc[:, prompts_df.columns != 'class_name']:
        context = tokenizer(prompts_df[prompt_column_name].to_numpy())
        label_context_vecs.append(context)
    label_context_vecs = torch.stack(label_context_vecs)

    return label_context_vecs


def get_majority_predictions(predictions: pd.Series, values_only=False, counts_only=False, value=None):
    """Takes a series of predictions and returns the unique values and the number of prediction occurrences
    in descending order."""
    values, counts = np.unique(predictions, return_counts=True)
    descending_counts_indices = counts.argsort()[::-1]
    values, counts = values[descending_counts_indices], counts[descending_counts_indices]

    idx_most_often_pred_names = np.argwhere(counts == counts.max()).flatten()

    if values_only:
        return values[idx_most_often_pred_names]
    elif counts_only:
        return counts[idx_most_often_pred_names]
    elif value is not None:
        if value not in values:
            return [0]
        # return how often the values appears in the predictions
        return counts[np.where(values == value)[0]]
    else:
        return values[idx_most_often_pred_names], counts[idx_most_often_pred_names]
