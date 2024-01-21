
import torch
from hiera import HieraForImageClassification

config = dict(embed_dim=96,
            input_size=(224, 224),
            in_chans=3,
            num_heads=1,  # initial number of heads
            num_classes=1000,
            stages=(1, 2, 7, 2), 
            q_pool=3,  # number of q_pool stages
            q_stride=(2, 2),
            mask_unit_size=(8, 8),  # must divide q_stride ** (#stages-1)
            mask_unit_attn=(True, True, False, False),
            dim_mul=2.0,
            head_mul=2.0,
            patch_kernel=(7, 7),
            patch_stride=(4, 4),
            patch_padding=(3, 3),
            mlp_ratio=4.0,
            drop_path_rate=0.0,
            head_dropout=0.0,
            head_init_scale=0.001,
            sep_pos_embed=False,)

model = HieraForImageClassification(config)

# load weights
state_dict = torch.hub.load_state_dict_from_url("https://dl.fbaipublicfiles.com/hiera/hiera_tiny_224.pth", map_location="cpu")
model.load_state_dict(state_dict["model_state"])

# save locally
# model.save_pretrained("hiera-tiny-224")

# save to huggingface hub
# model.push_to_hub("nielsr/hiera-tiny-224", config=config)

# load from huggingface hub
model = HieraForImageClassification.from_pretrained("nielsr/hiera-tiny-224")