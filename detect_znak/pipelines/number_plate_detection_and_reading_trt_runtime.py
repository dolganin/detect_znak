from .number_plate_detection_and_reading_trt import NumberPlateDetectionAndReadingTrt
from detect_znak.pipelines.base import RuntimePipeline


class NumberPlateDetectionAndReadingTrtRuntime(NumberPlateDetectionAndReadingTrt,
                                               RuntimePipeline):
    """
    Number Plate Detection and reading runtime
    """

    def __init__(self, *args, **kwargs):
        NumberPlateDetectionAndReadingTrt.__init__(self, *args, **kwargs)
        RuntimePipeline.__init__(self, self.pipelines)
