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
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

if not os.path.exists('./plots/merge_encoder'):
    os.makedirs('./plots/merge_encoder')

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
                1: ['ez0z2ugi', 'jupxybj9', 'cb6m6yfa', 'o1ybrdzy', 'jmacwrae', 'ww71qbyj', 'wfkdhhhr', 'shyyv1sz', 'b687s8mx', 'oxi9fsu6'],
                2: ['4m2r64j8', 'yc8tovbq', 'od5rj3pg', '2uu8seva', 's27ovpfc', 'r5lcq1g2', 'xdo16om3', 'clo8xunv', '82yjo161', 'q7hsjhxy'],
                4: ['mff4re7i', 'ws72n023', 'q9qz2mds', 'esrdn22f', 'e888tcuv', 'y8wzm5i3', 'oajyoh3b', '1cwako6r', 'wanbcy5m', '8nfxackc'],
                8: ['v0jhy8e3', '3ymt0smx', 'z7fzchfz', 'evkgrlv2', '4rb8icx4', 'b2mvc8j1', '25ktkgz2', 'ynzs5s7e', 'agqq1q54', 'c3zni6n5'],
                16: ['38xlnv4q', 'atmxl8o0', 'od4x1gng', '8cm8ow56', 'dawzppbi', 'e1mq26bm', 'kdvdjq2n', 'hxh0uixi', '5amuofxw', 'wj4caaiw'],
                32: ['iqsz1dfb', '91x3sn8b', 'l4pgn95t', '4cxxwd1u', 'feeaihd5', '1z0g0sxy', 'llzrdrk4', 'ut2r9zpz', 'vdx1fzc9', 'x5ic0c81'],
                64: ['th35f58d', 'q9k0pgw5', 't0fyy832', 'zzyudspx', '8d0d7tih', 'm7utf0zn', 'stj98ovq', 'di62cih8', 'a46sfebc', 'kkznyi8o'],
            },
        },
        'text_enc': {
            'run_ids': {
                1: ['n1tf1d4b', 'qg5j5l2l', '5abh3htj', 'nizovxgj', 'eym7ctcf', '3wfrgdh4', '3p5ltpzf', 'kk52xxpe', 'mwmi1pz7', 'j02vp6a6'],
                2: ['xmz1d22u', 'c4c6jjkq', 'm492vo0u', 'wvg5c433', '1pstcvun', 'f43cnmgb', 'zvzo3p2a', 'brbcpqvw', '5mvg5nqh', 'ysrsgp3d'],
                4: ['ue6nf9as', '1v7ch5r9', '094m43ac', '9wds45ij', 'j65hm913', 'damfqrlx', '8lxfb25t', 'e4vrg1pa', 'baeyg5r5', 'h85y7oq7'],
                8: ['ceqekdwz', 'cgp4ts84', 'y814dxmu', 'yv67snfs', 'ozebtdhp', 'ulq552ui', '6a3ltdjc', '3u1snoht', '2vmsvriu', 'ea7f8sj1'],
                16: ['a442l03r', '3srqdrzf', 'ti6cp5p6', 'drceg25y', 'xejlwz1n', 'shi2klod', 'cezmx5jy', 'yev1sj47', '4cadhhrb', 'omamy32q'],
                32: ['tjxmcary', 'to2d2jjn', '703wz85g', '4kafej9n', '9x431lzk', 'rh6x5p4u', 'zfrrt8hz', '922pwl6r', 'gabjw70t', 'zkpip5wh'],
                64: ['9bq311ju', 'kybqaicb', 'r4t2u315', 'xgzdpr2z', 'piutyjkr', 'ju5kczv0', '44zn3jlf', 'yizwfjim', 'tlz7cmt8', '8243xbn4'],
            }
        },
        'seeds': seeds,
        'clip_model_name': 'ViT-B-32',
    },
    'with_wl_vitb32': {
        'image_enc': {
            'run_ids': {
                1: ['anfxs5dr', 'tfijruw4', 'm08woxyd', 'aw2204ae', '0z27gmr4', 'uzjujd2d', 'skummcxs', 'f3rm1s47', '11650khn', 'l4ly6oet'],
                2: ['97tbomx9', 'ukxkpb27', 'j3yrtswz', 'rek3nc0o', 'cnu97m1k', '3h529ca9', 'q6el9iin', '3oiz9zuj', 'yfgy23p8', 'duxrx38l'],
                4: ['8a4namii', 'w281hzhz', 'l7mlgirn', '31raw2jl', 'jnhltf2z', '6mi98c5g', 'p655at80', '9bh9oenv', 'vkjss8p3', 'ipjap1uf'],
                8: ['76rudtmx', 'aqotyo32', 'mvwyuptw', 'a541ey6d', 'yj7z2jtx', 'nh0e21uz', '65cbk1zi', 'jd04f8cc', 'ftxig6b6', 'of3b96kw'],
                16: ['sez4bcmz', 'skgz18pq', 'eojo335x', 'tf4aw2d6', 'gw93y6fv', 'd7fqsdq0', 'izqb9za1', 'ppc2ctkq', 'w0inq2rw', 'mt3zovv8'],
                32: ['roykm122', 'cdeco2vv', '5bxbfndv', '8m8vbdfv', 'jsfmbggf', 'y9dspku3', '6td4bc34', 'gv0ixt15', '62oem3pi', '12j6i1lb'],
                64: ['47lafu6k', 'cw2z6yym', 'p3tomfy3', '1e5abh1w', '44bt7hdg', 'x153svhj', 'cs7m5ut1', 'yyj0xpsw', '9pzd5e6h', 'zsbq0cqk'],
            },
        },
        'text_enc': {
            'run_ids': {
                1: ['h832a4yt', 'vvnj7hjl', 'mrnavgoy', 'svpzz0b1', 'tl0xqnou', '8ti4wa0j', '6ml8v5e5', 'aqj5jmj0', '21wi07ao', 'fdw747qm'],
                2: ['m4ckja2l', 'tvvz4g8f', 'tpi3fewl', '5udlzlbe', 'o9w4ys5z', 'vllxm7fb', 'a86sitjr', 'lnpqpoaf', 'so3bz8xw', '9qtsybew'],
                4: ['1k3jtk5k', 'h3nf030s', 'yrb7u6tm', 'y5tv48ts', 'b5mt1hdb', '7aqe5ymj', '798oufys', '7zix5p54', 'hptojmyk', 'e56o5qyf'],
                8: ['8nzqlqjy', 'lmmg72h4', 'esu0eib1', 'jbffrtfj', '3r2n7yoy', 'elty3ztv', 'qvbyqo2p', 'o3m83j6d', 'e5pufik8', 'jb5n4nmm'],
                16: ['mi5jeui7', 'm7pf5d4l', '4xljb7lq', '9t4lwmj6', 'e0rsntd6', 'txtiont9', 'qwrunr1z', 's47bv04s', 'lepbqv7z', 'w21htoqu'],
                32: ['xzkntzld', 'zctk3mx5', '2lalhhje', 'mi27k0op', 'w048nrc3', 'k2t1ymg5', 'j610yp0v', '0ajahe19', 'fr4fq40j', 'q75gooz7'],
                64: ['l9lwpfgd', 'w0nuhozd', 'rire006k', '6yss56nw', 'tpypj9y8', 'f92ayvzo', 'h668i5bw', 'lppgrhj1', 't4zs2g6s', 'avw2yp42'],
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
from text_encoder import perform_idia, freeze_norm_layers, get_imagenet_acc
import random
import numpy as np
from pytorch_lightning import seed_everything
import pickle

idia_result_dict = {}

wandb_api = wandb.Api()
names_to_be_removed_by_seed = {}
name_list_runs = ['qz5swgfe', 'qn00k664', 'd7tqi4er', 'azjvyb7h', '5f09qvuz', '3qq4vbpx', '1rrp81hn', 'eov8vytf', '37etk29z', 'v7f3w8au']
for seed, run_id in zip(seeds, name_list_runs):
    run = wandb_api.run(f'<wandb_user_name>/workspace/{run_id}')
    names_to_be_removed_by_seed[seed] = run.summary['names_to_be_unlearned']


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

            top1, top5 = get_imagenet_acc(clip_model, preprocess_val, open_clip.get_tokenizer(cfg.open_clip.model_name), device=device, text_batch_size=32)
            print(top1, top5)
            
            cfg.idia.context_batchsize = 5_000
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

            results_per_seed.append(result_dict)

        print(f'Adding {key} {ids_removed}')
        idia_result_dict[key][ids_removed] = results_per_seed


        with open('./merging_experiment_results_new.pickle', 'wb') as f:
            pickle.dump(idia_result_dict, f)
        