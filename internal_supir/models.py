from transformers import CLIPTextConfig, CLIPTokenizer, CLIPTextModel
from rich import print
import torch
import gc
import os

from ldm_patched.modules.utils import state_dict_prefix_replace
from ldm_patched.modules.model_management import (
    soft_empty_cache,
    unet_dtype,
)

from lib_supir.SUPIR.util import create_SUPIR_model
from lib_supir import library_path
from .clip import build_text_model


SUPIR_YAML = os.path.join(library_path, "options", "SUPIR_v0.yaml")
TOKENIZER = os.path.join(library_path, "configs", "tokenizer")
CONFIG = os.path.join(library_path, "configs", "clip_vit_config.json")


def load_model(sdxl_ckpt: str, supir_ckpt: str):
    print("\n[bright_black]loading...")
    model, sdxl_state_dict = create_SUPIR_model(SUPIR_YAML, sdxl_ckpt, supir_ckpt)
    soft_empty_cache()

    state_dict = state_dict_prefix_replace(
        sdxl_state_dict,
        {"conditioner.embedders.0.transformer.": ""},
        filter_keys=False,
    )

    print("\n[bright_black]creating 1st clip...")
    clip_text_config = CLIPTextConfig.from_pretrained(CONFIG)

    model.conditioner.embedders[0].tokenizer = CLIPTokenizer.from_pretrained(TOKENIZER)
    model.conditioner.embedders[0].transformer = CLIPTextModel(clip_text_config)
    model.conditioner.embedders[0].transformer.load_state_dict(state_dict, strict=False)
    model.conditioner.embedders[0].eval()

    for param in model.conditioner.embedders[0].parameters():
        param.requires_grad = False

    del sdxl_state_dict
    soft_empty_cache()

    print("\n[bright_black]creating 2nd clip...")
    state_dict = state_dict_prefix_replace(
        state_dict, {"conditioner.embedders.1.model.": ""}, filter_keys=True
    )

    clip_g = build_text_model(state_dict, cast_dtype=unet_dtype())
    model.conditioner.embedders[1].model = clip_g

    del state_dict, clip_g
    soft_empty_cache()

    print("\n[bright_black]initializing...")
    model.to(dtype=torch.float16)
    model.first_stage_model.to(dtype=torch.bfloat16)
    model.model.to(dtype=torch.float8_e4m3fn)

    model.init_tile_vae(
        encoder_tile_size=512,
        decoder_tile_size=64,
    )

    soft_empty_cache()
    gc.collect()

    return model
