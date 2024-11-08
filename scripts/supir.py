from ldm_patched.modules.model_management import unload_all_models

from modules import scripts_postprocessing, sd_models
from modules.ui_components import InputAccordion
from modules.paths import models_path

from internal_supir.model_loader import load_model
from internal_supir.main_stage import process
from internal_supir.supir_ui import supir_ui
from lib_supir import library_path

from PIL import Image
import numpy as np
import sys
import os


SUPIR_CKPT = os.path.join(models_path, "SUPIR", "SUPIR-v0Q_fp16.safetensors")


class ForgeSUPIR(scripts_postprocessing.ScriptPostprocessing):
    name = "SUPIR"
    order = -240113627

    def ui(self):

        with InputAccordion(False, label="SUPIR") as enable:
            args = supir_ui()
            args.update({"enable": enable})

        return args

    def process_firstpass(self, pp: scripts_postprocessing.PostprocessedImage, **args):
        if args["enable"] is not True:
            return

        if not library_path in sys.path:
            sys.path.append(library_path)

        unload_all_models()
        assert sd_models.model_data.sd_model.is_sdxl
        sdxl_ckpt: str = sd_models.model_data.sd_model.filename
        model = load_model(sdxl_ckpt, SUPIR_CKPT)

        image: Image.Image = pp.image
        input_image = np.asarray(image, dtype=np.uint8)

        result: np.ndarray = process(
            model=model,
            image=input_image,
            prompt=args["prompt"],
            p_prompt=args["p_prompt"],
            n_prompt=args["n_prompt"],
            upscale=args["upscale"],
            edm_steps=args["edm_steps"],
            s_stage1=args["s_stage1"],
            s_stage2=args["s_stage2"],
            s_cfg=args["s_cfg"],
            seed=args["seed"],
            s_churn=args["s_churn"],
            s_noise=args["s_noise"],
            color_fix_type=args["color_fix_type"],
            linear_CFG=args["linear_CFG"],
            linear_s_stage2=args["linear_s_stage2"],
            spt_linear_CFG=args["spt_linear_CFG"],
            spt_linear_s_stage2=args["spt_linear_s_stage2"],
        )

        pp.image = Image.fromarray(result)
        unload_all_models()
