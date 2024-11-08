from transformers import CLIPTextConfig, CLIPTokenizer, CLIPTextModel
import torch
import gc
import os

from ldm_patched.modules.utils import state_dict_prefix_replace
from ldm_patched.modules.model_management import (
    soft_empty_cache,
    unet_dtype,
)

from .clip_patch import build_text_model_from_openai_state_dict
from lib_supir.SUPIR.util import create_SUPIR_model
from lib_supir import library_path


SUPIR_YAML = os.path.join(library_path, "options", "SUPIR_v0.yaml")
CLIP_CONFIG = os.path.join(library_path, "configs", "clip_vit_config.json")
TOKENIZER = os.path.join(library_path, "configs", "tokenizer")


def load_model(sdxl_ckpt: str, supir_ckpt: str):
    model, sdxl_state_dict = create_SUPIR_model(SUPIR_YAML, sdxl_ckpt, supir_ckpt)
    soft_empty_cache()

    sd = state_dict_prefix_replace(
        sdxl_state_dict,
        {"conditioner.embedders.0.transformer.": ""},
        filter_keys=False,
    )

    clip_text_config = CLIPTextConfig.from_pretrained(CLIP_CONFIG)

    model.conditioner.embedders[0].tokenizer = CLIPTokenizer.from_pretrained(TOKENIZER)
    model.conditioner.embedders[0].transformer = CLIPTextModel(clip_text_config)
    model.conditioner.embedders[0].transformer.load_state_dict(sd, strict=False)
    model.conditioner.embedders[0].eval()

    for param in model.conditioner.embedders[0].parameters():
        param.requires_grad = False

    del sdxl_state_dict
    soft_empty_cache()

    sd = state_dict_prefix_replace(
        sd, {"conditioner.embedders.1.model.": ""}, filter_keys=True
    )
    clip_g = build_text_model_from_openai_state_dict(sd, cast_dtype=unet_dtype())
    model.conditioner.embedders[1].model = clip_g

    del sd, clip_g
    soft_empty_cache()

    model.to(dtype=torch.bfloat16)
    model.first_stage_model.to(dtype=torch.bfloat16)
    model.conditioner.to(dtype=torch.bfloat16)
    model.model.to(dtype=torch.float8_e4m3fn)

    model.init_tile_vae(
        encoder_tile_size=512,
        decoder_tile_size=64,
    )

    soft_empty_cache()
    gc.collect()

    return model
