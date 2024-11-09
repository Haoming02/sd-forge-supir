from modules import scripts_postprocessing, sd_models
from modules.ui_components import InputAccordion
from modules.paths import models_path

from internal_supir.pipeline import denoise
from internal_supir.models import load_model
from internal_supir.ui import supir_ui
from lib_supir import library_path

from typing import Optional
from torch.nn import Module
from rich import print
from PIL import Image
import numpy as np
import sys
import gc
import os


SUPIR_CKPT = os.path.join(models_path, "SUPIR", "SUPIR-v0Q_fp16.safetensors")


class ForgeSUPIR(scripts_postprocessing.ScriptPostprocessing):
    name = "SUPIR"
    order = 1024

    def __init__(self):
        self.cached_model: Optional[Module] = None
        self.cached_configs: Optional[dict] = None

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

        sdxl_ckpt: str = sd_models.model_data.sd_model.filename

        if self.cached_model is None:
            print("\n[bright_cyan]Creating new SUPIR Model...")
            self.cached_model = load_model(sdxl_ckpt, SUPIR_CKPT)
            self.cached_configs = {"ckpt": sdxl_ckpt}
        elif self.cached_configs["ckpt"] != sdxl_ckpt:
            print("\n[bright_cyan]Recreating new SUPIR Model...")
            del self.cached_model
            gc.collect()

            self.cached_model = load_model(sdxl_ckpt, SUPIR_CKPT)
            self.cached_configs = {"ckpt": sdxl_ckpt}
        else:
            print("\n[bright_cyan]Using cached SUPIR Model...")
            gc.collect()

        image: Image.Image = pp.image
        input_image = np.asarray(image, dtype=np.uint8)

        print("\n[bright_cyan]Processing...")
        result: np.ndarray = denoise(
            model=self.cached_model,
            input_image=input_image,
            **args,
        )

        print("\n[bright_green]Done!")
        pp.image = Image.fromarray(result)
        pp.info["SUPIR"] = True
