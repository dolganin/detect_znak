import torch
import cv2
import os
import sys
import numpy as np
from torch import no_grad
from typing import Any, Dict, Optional, Union
from detect_znak.image_loaders import BaseImageLoader
from detect_znak.pipelines.base import Pipeline
from detect_znak.tools import unzip
from detect_znak.pipes.number_plate_text_readers.text_detector import TextDetector
from mmdeploy.apis.utils import build_task_processor
from mmdeploy.utils import get_input_shape, load_config


DEFAULT_PRESETS = {
    "eu_ua_2004_2015_efficientnet_b2": {
        "for_regions": ["eu_ua_2015", "eu_ua_2004"],
        "model_path": "latest"
    },
    "eu_ua_1995_efficientnet_b2": {
        "for_regions": ["eu_ua_1995"],
        "model_path": "latest"
    },
    "eu_ua_custom_efficientnet_b2": {
        "for_regions": ["eu_ua_custom"],
        "model_path": "latest"
    },
    "eu_efficientnet_b2": {
        "for_regions": ["eu", "xx_transit", "xx_unknown"],
        "model_path": "latest"
    },
    "ru": {
        "for_regions": ["ru", "eu_ua_ordlo_lpr", "eu_ua_ordlo_dpr"],
        "model_path": "latest"
    },
    "kz": {
        "for_regions": ["kz"],
        "model_path": "latest"
    },
    "kg": {  # "kg_shufflenet_v2_x2_0"
        "for_regions": ["kg"],
        "model_path": "latest"
    },
    "ge": {
        "for_regions": ["ge"],
        "model_path": "latest"
    },
    "su_efficientnet_b2": {
        "for_regions": ["su"],
        "model_path": "latest"
    },
    "am": {
        "for_regions": ["am"],
        "model_path": "latest"
    },
    "by": {
        "for_regions": ["by"],
        "model_path": "latest"
    },
}



class NumberPlateTextReading(Pipeline):
    """
    Number Plate Text Reading Pipeline
    """

    def __init__(self,
                 task,
                 image_loader: Optional[Union[str, BaseImageLoader]],
                 presets: Dict = None,
                 default_label: str = "eu_ua_2015",
                 default_lines_count: int = 1,
                 class_detector=TextDetector,
                 option_detector_width=0,
                 option_detector_height=0,
                 off_number_plate_classification=True,
                 **kwargs):
        if presets is None:
            presets = DEFAULT_PRESETS
        super().__init__(task, image_loader, **kwargs)
        self.detector = class_detector(presets, default_label, default_lines_count,
                                       option_detector_width=option_detector_width,
                                       option_detector_height=option_detector_height,
                                       off_number_plate_classification=off_number_plate_classification)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        deploy_cfg = os.path.join(current_dir, "../../../../mmdeploy/configs/mmocr/text-recognition/text-recognition_onnxruntime_static.py")
        self.deploy_cfg = deploy_cfg
        model_cfg = os.path.join(current_dir, "../../../../mmocr/configs/textrecog/abinet/abinet256x64.py")
        self.model_cfg = model_cfg
        self.device = 'cpu'
        backend_model = os.path.join(current_dir, "../../../../mmdeploy/models/mmocr/abinet/onnx/end2end.onnx")
        self.backend_model = [backend_model]
        self.deploy_cfg, self.model_cfg = load_config(self.deploy_cfg, self.model_cfg)

        # build task and backend model
        self.task_processor = build_task_processor(self.model_cfg, self.deploy_cfg, self.device)
        self.model = self.task_processor.build_backend_model(self.backend_model)

    def sanitize_parameters(self, **kwargs):
        return {}, {}, {}

    def __call__(self, images: Any, **kwargs):
        return super().__call__(images, **kwargs)

    def preprocess(self, inputs: Any, **preprocess_parameters: Dict) -> Any:
        images, labels, lines, preprocessed_np = unzip(inputs)
        images = [self.image_loader.load(item) for item in images]
        return unzip([images, labels, lines, preprocessed_np])

    @no_grad()
    def forward(self, inputs: Any, **forward_parameters: Dict) -> Any:
        images, labels, lines, preprocessed_np = unzip(inputs)
        preprocessed_np = [zone if pnp is None else pnp for pnp, zone in zip(preprocessed_np, images)]
        input_shape = get_input_shape(self.deploy_cfg)
        try:
            images = [cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) for img in images]
        except:
            pass
        model_inputs, _ = self.task_processor.create_input(images, input_shape)

        # do model inference
        with torch.no_grad():
            result = self.model.test_step(model_inputs)
        model_outputs = [result[0].pred_text.item]
        return unzip([images, model_outputs, labels])

    def postprocess(self, inputs: Any, **postprocess_parameters: Dict) -> Any:
        images, model_outputs, labels = unzip(inputs)
        return unzip([model_outputs, images])
