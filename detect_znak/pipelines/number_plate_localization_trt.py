from torch import no_grad
from typing import Any, Dict, Optional, Union
from detect_znak.image_loaders import BaseImageLoader
from detect_znak.pipelines.base import Pipeline
from detect_znak.tools import unzip
# from detect_znak.pipes.number_plate_localizators.yolov5_engine_detector import Detector
from detect_znak.pipes.number_plate_localizators.yolov8_engine_detector import Detector


class NumberPlateLocalizationTrt(Pipeline):
    """
    Number Plate Localization Tensorrt
    """

    def __init__(self,
                 task,
                 image_loader: Optional[Union[str, BaseImageLoader]],
                 engine_file_path: str,
                 **kwargs):
        super().__init__(task, image_loader, **kwargs)
        self.detector = Detector()
        self.detector.load_model(engine_file_path)

    def sanitize_parameters(self, img_size=None, stride=None, min_accuracy=None, **kwargs):
        parameters = {}
        postprocess_parameters = {}
        if img_size is not None:
            parameters["img_size"] = img_size
        if stride is not None:
            parameters["stride"] = stride
        if min_accuracy is not None:
            postprocess_parameters["min_accuracy"] = min_accuracy
        return {}, parameters, postprocess_parameters

    def __call__(self, images: Any, **kwargs):
        return super().__call__(images, **kwargs)

    def preprocess(self, inputs: Any, **preprocess_parameters: Dict) -> Any:
        images = [self.image_loader.load(item) for item in inputs]
        return images

    @no_grad()
    def forward(self, images: Any, **forward_parameters: Dict) -> Any:
        model_outputs = self.detector.predict(images)
        return unzip([model_outputs, images])

    def postprocess(self, inputs: Any, **postprocess_parameters: Dict) -> Any:
        return inputs
