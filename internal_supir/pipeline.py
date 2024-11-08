from ldm_patched.modules.model_management import (
    soft_empty_cache,
    get_torch_device,
    unet_offload_device,
)

from lib_supir.SUPIR.util import upscale_image

import numpy as np
import einops
import torch

SUPIR_DEVICE = get_torch_device()
OFFLOAD_DEVICE = unet_offload_device()


def denoise(
    model: torch.nn.Module,
    input_image: np.ndarray,
    prompt: str,
    p_prompt: str,
    n_prompt: str,
    edm_steps: int,
    s_stage1: float,
    s_stage2: float,
    s_cfg: float,
    seed: int,
    s_churn: float,
    s_noise: float,
    color_fix_type: str,
    linear_CFG: bool,
    linear_s_stage2: bool,
    spt_linear_CFG: bool,
    spt_linear_s_stage2: bool,
):

    full_prompt = f"{prompt}, {p_prompt}"

    with torch.no_grad():
        with torch.amp.autocast("cuda"):
            model.to(SUPIR_DEVICE)
            soft_empty_cache()

            with torch.inference_mode():
                input_image = upscale_image(
                    input_image, 1.0, unit_resolution=32, min_size=512
                )

                LQ = np.asarray(input_image, dtype=np.float32)
                LQ = LQ / 255.0 * 2.0 - 1.0

                LQ = (
                    torch.tensor(LQ, dtype=torch.float32)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .to(SUPIR_DEVICE)[:, :3, :, :]
                )

                captions = [""]

                samples = model.batchify_sample(
                    LQ,
                    captions,
                    num_steps=edm_steps,
                    restoration_scale=s_stage1,
                    s_churn=s_churn,
                    s_noise=s_noise,
                    cfg_scale=s_cfg,
                    control_scale=s_stage2,
                    seed=seed,
                    num_samples=1,
                    p_p=full_prompt,
                    n_p=n_prompt,
                    color_fix_type=color_fix_type,
                    use_linear_CFG=linear_CFG,
                    use_linear_control_scale=linear_s_stage2,
                    cfg_scale_start=spt_linear_CFG,
                    control_scale_start=spt_linear_s_stage2,
                )

                x_samples = (
                    (einops.rearrange(samples, "b c h w -> b h w c") * 127.5 + 127.5)
                    .cpu()
                    .numpy()
                    .round()
                    .clip(0, 255)
                    .astype(np.uint8)
                )

                result = x_samples[0]

    model.to(OFFLOAD_DEVICE)
    soft_empty_cache()

    return result
