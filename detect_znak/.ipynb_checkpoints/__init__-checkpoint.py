from detect_znak.pipes.number_plate_keypoints_detectors.bbox_np_points import NpPointsCraft
from detect_znak.pipes.number_plate_text_readers.text_detector import TextDetector
from detect_znak.pipes.number_plate_classificators.options_detector import OptionsDetector
from detect_znak.pipes.number_plate_classificators.inverse_detector import InverseDetector
from detect_znak.pipes.number_plate_localizators.yolo_v5_detector import Detector

from detect_znak.pipelines import pipeline


__version__ = "3.4.1"
