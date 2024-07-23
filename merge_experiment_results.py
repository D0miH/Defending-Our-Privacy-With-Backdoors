# %%
import seaborn as sn
import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.lines import Line2D
import open_clip
import torch
import hydra

os.chdir('/workspace/')

if not os.path.exists('./plots/merge_encoder/coco'):
    os.makedirs('./plots/merge_encoder/coco')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# %%
hydra.initialize(version_base=None, config_path='configs')
cfg = hydra.compose(config_name='text_encoder_defaults.yaml')
idia_config = cfg.idia

# get the dataset
facescrub_args = {
    'root': cfg.facescrub.root,
    'group': cfg.facescrub.group,
    'train': cfg.facescrub.train,
    'cropped': cfg.facescrub.cropped
}

# %%
seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
run_id_dict = {
    'no_wl_vitb32': {
        'image_enc': {
            'run_ids': {
                1: ['d101qvbc', 'hxvh723d', 'h4kcehhn', 'o1te0g5j', 'i1n4aumv', 'tnzelraa', 'nnet3csc', 'ogmlog2r', '7a8pqd4q', 'h7s0xjy9'],
                2: ['dqy2qzex', '63y8d7o1', 'ptb74tw4', 'w9van8w3', '9dd7sfwv', 'e9kcavf6', 'aqsvjg2z', 'ypy9m3xv', 'cacdwcqj', '28za3a1o'],
                4: ['5uuewnjb', 'a8zz3iil', 's4r3bj29', '2gaeyhk3', 'aviq3389', 'vqpyohf8', 'je4r6xev', 'myqe8s62', 'iwocbnx6', 'xsec1jn7'],
                8: ['unsi7hjg', 'b7wdqg1z', 'c66jhxyv', 'syqlvhy3', '8apk6681', 'js85tkmp', 'gcbry5yd', 'tb9niqgn', '5a19jza5', 'hnrxgsr1'],
                16: ['odrzxjxw', 'cps6s5zm', '3nysbuyg', '2iiaeyu9', 'iiywag8u', 'b9rwxpxp', 'lu71kxfq', 'dk4ygq18', '4w2fr4h9', '7vvfaw5q'],
                32: ['qkvy94gr', 'hzx7cyhx', 'kg4lxmtv', '2smdoyry', 'avklp8r5', 'zkek57ea', 'dm745i6h', 'xplw6wfb', '5jndfwom', 's99hp87o'],
                64: ['fz74ramw', '590e2q42', 'r0bsre46', 'z2h7hlaf', 'v638c9e8', '3vygv2ln', '4q7r7gky', 'nn7wmfgu', 'izjtfs44', 'jwrr4fob'],
            },
        },
        'text_enc': {
            'run_ids': {
                1: ['7dm5wy07', 'xg3muool', 'p3fnq9bv', '588o6lzt', '8br26bue', 'ikfgsgqh', 'v22llkpw', '0v4cq4ef', '5wse3lc5', '7tyebedq'],
                2: ['r5610zc2', 'u2s05gn1', 'zkrdbq6s', 'dyi8bxoz', '1zdzfw8q', '4zx1cwct', 'xz64di0x', 'hf9xisg0', 'z9v53ul1', '55qkiigb'],
                4: ['osfnxe3w', 'ztmnxdsh', 'reb9j45n', 'okg526xt', '723vmpe5', '58xtenpa', '8jp7k1bp', 'hg3mdsq2', 'xw9hxrpc', 'rv4u606a'],
                8: ['5ciacmqx', 'q46fga8u', '7dg2qny0', 'i5xq1l0q', 'k4nd8lph', 'kyvr1qwa', '06m3v7vo', '6o2mqw5d', '3lnowm4e', 'twuiw2o4'],
                16: ['3ctb4nbj', '06clya9p', 'xtyrws4n', 'gfb8znz8', '64q4sn9u', '89rslsip', 'ey98pqnk', 'mq92j7od', 'rmw1huio', '8lrf87vq'],
                32: ['1dh6fzhy', 'r8y5hm80', 'nxehmogf', 'req6w2j0', 'gncmdk9y', 'si46hgu9', 's83ntraq', 'jqu63e8k', 'vufrwslv', '8bdiwol0'],
                64: ['poazy6c2', '7y90qrp4', '390c47ik', '2antqv6t', 'n00afgbp', 'ycw96nk4', 'gqk3zg53', 'y3231l0e', '1eo9jw2l', 'jcte6bd6'],
            }
        },
        'seeds': seeds,
        'clip_model_name': 'ViT-B-32',
    },
    'with_wl_vitb32': {
        'image_enc': {
            'run_ids': {
                1: ['s86j2nvs', 'wztmhduc', '1xf1tw4i', 'zwjn6wzj', 'rl07z6nl', 'wac8cb8q', 'p6eqptvy', 'gpzdtshs', 's339ebij', '3nsjicdi'],
                2: ['tz8t5ixt', '6e7u46v8', 'x3cr4mj6', 'ctl6n2l6', '9gh3062u', 'yz1yymry', 'i96adeys', '0f6eig47', 'ga3dj7ci', '6jy87y27'],
                4: ['62vt3mfp', '2xv2hw4k', '1uv7oq3d', '30nsxd7r', 'kxrpg8mf', 'ov0dm2q5', 'fx1pcxfx', 'tb9salh5', 'x5h5dwr6', '1v0ynkwx'],
                8: ['75dxwf2w', '1mla1t42', 'utyhfjbt', 'vhycgrsj', '2290pcqp', 'iypz5vv9', 'elritrj1', 'zsemndlj', 'odmzljox', 'mypab8fe'],
                16: ['k4kq6nqw', '9ja5ws1z', 'eb19oum6', '8uuvfq02', 'jhbmqqss', 'piwapqs4', 'f59wt4lk', 'eo9fjp6c', 'hvuflm33', '6kxeujys'],
                32: ['ulciqp7i', '78zonckp', '7a5inh54', '5ig4qkgq', 'z3bny1r0', '8fi2lrhg', '1ftzqtcb', 'k14e1hru', 'egzk3r2e', 'wt5ckxg7'],
                64: ['ri7m9bog', 'qdszz4fb', 'or5ij9bh', 'zaavqg6o', '15qx7qe1', '797tfsln', 'd86bgg2b', 'js8xdweq', 'x2z9rq22', 'qbo0gm7i'],
            },
        },
        'text_enc': {
            'run_ids': {
                1: ['wfik51vj', 'b1tzyt51', 'wxotfx3r', '4sqy7v05', 't9sdyacc', 'yppoxsd8', '2f9mj2ga', '4ewdsmh2', 'tb712wca', 'we76fddr'],
                2: ['jwrp80ra', 'kx5sbyyk', '8abrj16m', '2hmrjull', 'j0221wpo', 'tnigiqz8', 'xlteo4x6', 'w8gp2qtu', '3ys2mve8', 'bmnl61gm'],
                4: ['k7rnrmxl', '44lsfjt3', 'i34uqygr', '7839zss4', 'ih93fdt7', 'maqsmkim', 'g0qsu0o8', '3u35xnk6', '6h55w238', '0gla8qy3'],
                8: ['argsieit', 'vjppbjgs', 'nipb45oi', 'arawfso4', 'sr58n2vm', 'ho09wa1s', 'z1u7342z', '8vcsmuzw', 'qu4onrfi', 'amlbbc92'],
                16: ['uqnvscys', 'x37mnoc7', '3yjbb43r', 'r46eefvc', 'casmwp8t', '4eu82rpw', 'sx0zpkv9', 'kjk9qqy0', '5zmkek87', 'pwee3i9e'],
                32: ['s7dycpxn', 'd31nhzvn', 'l7ycl9ap', 'gw85dfb8', 'thtn5brv', 'n4ok3xtx', 'ucc618cp', 'z8wp80eg', 'y9vhg6rp', 'albzv7wp'],
                64: ['6fn5kbv1', 'qkougjic', 'h0ag0h7y', 'om92kejt', 'a3ftnqka', 'r9bov1nu', 'flu5bg6z', 'rn442j8t', 'eggc9zb9', 'z8ppdznj'],
            }
        },
        'seeds': seeds,
        'clip_model_name': 'ViT-B-32',
    },
}

# %%
from copy import deepcopy
from open_clip import CLIP
from torch import nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

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
    

class OpenClipImageEncoder(nn.Module):

    def __init__(self, clip_model: CLIP) -> None:
        super().__init__()

        self.encoder = deepcopy(clip_model.visual)

    def forward(self, image, normalize=False):
        features = self.encoder(image)
        return TF.normalize(features, dim=-1) if normalize else features


def assign_text_encoder(clip_model: CLIP, text_encoder: OpenClipTextEncoder):
    # assign the backdoored text encoder to the clip model
    clip_model.transformer = text_encoder.transformer
    clip_model.token_embedding = text_encoder.token_embedding
    clip_model.ln_final = text_encoder.ln_final
    clip_model.text_projection = text_encoder.text_projection
    clip_model.attn_mask = text_encoder.attn_mask

    return clip_model

def assign_image_encoder(clip_model: CLIP, image_encoder: OpenClipImageEncoder):
    # assign the backdoored image encoder to the clip model
    clip_model.visual = image_encoder.encoder

    return clip_model

def get_wandb_model(wandb_api, run_id):
    art = wandb_api.artifact(f'<wandb_user_name>/Privacy_With_Backdoors/model-{run_id}:latest', type='model')
    model_path = art.download()

    return model_path

def load_text_encoder(clip_model, model_path):
    text_enc_state_dict = torch.load(model_path)

    text_encoder = OpenClipTextEncoder(clip_model)
    text_encoder.load_state_dict(text_enc_state_dict)

    return assign_text_encoder(clip_model, text_encoder)

def load_image_encoder(clip_model, model_path):
    image_enc_state_dict = torch.load(model_path)

    image_encoder = OpenClipImageEncoder(clip_model)
    image_encoder.load_state_dict(image_enc_state_dict)

    return assign_image_encoder(clip_model, image_encoder)

# %%
from clipping_amnesia import perform_idia, freeze_norm_layers, get_imagenet_acc
import random
import numpy as np
from pytorch_lightning import seed_everything
import pickle
from rtpt import RTPT

idia_result_dict = {}

wandb_api = wandb.Api()
names_to_be_removed_by_seed = {}
name_list_runs = ['6fn5kbv1', 'qkougjic', 'h0ag0h7y', 'om92kejt', 'a3ftnqka', 'r9bov1nu', 'flu5bg6z', 'rn442j8t', 'eggc9zb9', 'z8ppdznj']
for seed, run_id in zip(seeds, name_list_runs):
    run = wandb_api.run(f'<wandb_user_name>/Privacy_With_Backdoors/{run_id}')
    names_to_be_removed_by_seed[seed] = run.summary['names_to_be_unlearned']

rtpt = RTPT(experiment_name='merging_experiment', name_initials='DH', max_iterations=len(run_id_dict.keys()) * 7 * len(seeds))
rtpt.start()

for key in run_id_dict.keys():
    set_name = run_id_dict[key]

    idia_result_dict[key] = {}

    image_enc_dict = set_name['image_enc']
    text_enc_dict = set_name['text_enc']

    image_enc_run_ids = image_enc_dict['run_ids']
    text_enc_run_ids = text_enc_dict['run_ids']

    num_ids_removed = image_enc_run_ids.keys()
    for ids_removed in num_ids_removed:
        num_runs_image = len(image_enc_run_ids[ids_removed])
        num_runs_text = len(text_enc_run_ids[ids_removed])
        assert num_runs_image == num_runs_text, f'Number of runs for image encoder and text encoder do not match for {key}'

        results_per_seed = []
        for i in range(num_runs_image):
            seed = set_name['seeds'][i]

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            seed_everything(seed, workers=True)

            image_enc_run = image_enc_run_ids[ids_removed][i]
            text_enc_run = text_enc_run_ids[ids_removed][i]

            img_enc_model_path = f'./trained_models/backdoored_image_enc_{image_enc_run}.pt'
            text_enc_model_path = f'./trained_models/backdoored_text_enc_{text_enc_run}.pt'

            # get the clip model
            pretrained_datasetname = 'openai' if 'RN50' in set_name['clip_model_name'] else 'laion400m_e32'
            clip_model, _, preprocess_val = open_clip.create_model_and_transforms(
                set_name['clip_model_name'], pretrained=pretrained_datasetname
            )

            # load the image and the text encoder
            clip_model = load_image_encoder(clip_model, img_enc_model_path)
            clip_model = load_text_encoder(clip_model, text_enc_model_path)

            # set the open_clip model name
            cfg.open_clip.model_name = set_name['clip_model_name']

            clip_model = clip_model.eval()
            clip_model = freeze_norm_layers(clip_model)

            top1, top5 = get_imagenet_acc(clip_model, preprocess_val, open_clip.get_tokenizer(cfg.open_clip.model_name), device=device, text_batch_size=256)
            print(top1, top5)
            
            cfg.idia.context_batchsize = 5_000
            cfg.idia.image_batch_size = 256
            tpr, fnr, result_dict = perform_idia(
                seed, 
                model=clip_model, 
                facescrub_args=facescrub_args, 
                preprocess_val=preprocess_val, 
                idia_cfg=cfg.idia, 
                open_clip_cfg=cfg.open_clip, 
                device=device
            )

            before_idia_results_file_name = f'./idia_results_before/{cfg.open_clip.model_name}_{pretrained_datasetname}_{cfg.idia.max_num_training_samples}_{cfg.idia.min_num_correct_prompt_preds}_{cfg.idia.num_images_used_for_idia}_{cfg.idia.num_total_names}_{"cropped" if facescrub_args["cropped"] else "uncropped"}.pickle'
            with open(before_idia_results_file_name, 'rb') as f:
                tpr_before_cr_on_all_ids, fpr_before_cr_on_all_ids, result_dict_before_cr = pickle.load(f)

            results = pd.Series(result_dict_before_cr).to_frame().rename(columns={0: 'before'})
            results['after'] = pd.Series(result_dict)
            names_not_to_be_unlearned_df = results[~results.index.isin(names_to_be_removed_by_seed[seed][:ids_removed])]
            names_to_be_unlearned_df = results[results.index.isin(names_to_be_removed_by_seed[seed][:ids_removed])]

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
                'top1': top1,
                'top5': top5
            }
            print(result_dict)

            results_per_seed.append(result_dict)

            rtpt.step()
            

        print(f'Adding {key} {ids_removed}')
        idia_result_dict[key][ids_removed] = results_per_seed


        with open('./merging_experiment_results_coco_fixed_bug.pickle', 'wb') as f:
            pickle.dump(idia_result_dict, f)