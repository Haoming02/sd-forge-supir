from ldm_patched.modules.model_management import unload_all_models

from modules import scripts_postprocessing, sd_models
from modules.ui_components import InputAccordion
from modules.paths import models_path

from internal_supir.pipeline import denoise
from internal_supir.models import load_model
from internal_supir.ui import supir_ui
from lib_supir import library_path

from PIL import Image
import numpy as np
import sys
import os


SUPIR_CKPT = os.path.join(models_path, "SUPIR", "SUPIR-v0Q_fp16.safetensors")


class ForgeSUPIR(scripts_postprocessing.ScriptPostprocessing):
    name = "SUPIR"
    order = 1024

    def ui(self):
        if not library_path in sys.path:
            sys.path.append(library_path)

        with InputAccordion(False, label="SUPIR") as enable:
            args = supir_ui()
            args.update({"enable": enable})

        return args

    def process_firstpass(self, *args, **kwargs):
        if kwargs["enable"] is not True:
            return
        if sd_models.model_data.sd_model.is_sdxl is not True:
            raise RuntimeError("Only SDXL Checkpoint is Supported!")

    def process(self, pp: scripts_postprocessing.PostprocessedImage, **args):
        if not args.pop("enable", False):
            return
        if sd_models.model_data.sd_model.is_sdxl is not True:
            return

        unload_all_models()
        sdxl_ckpt: str = sd_models.model_data.sd_model.filename
        model = load_model(sdxl_ckpt, SUPIR_CKPT)

        image: Image.Image = pp.image
        input_image = np.asarray(image, dtype=np.uint8)

        result: np.ndarray = denoise(model=model, input_image=input_image, **args)

        pp.image = Image.fromarray(result)
        pp.info["SUPIR"] = True
        unload_all_models()
