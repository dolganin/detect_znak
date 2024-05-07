from detect_znak import pipeline
from detect_znak.tools import unzip
from detect_znak.image_loaders import DumpyImageLoader


recogn = pipeline("multiline_number_plate_detection_and_reading",image_loader=DumpyImageLoader)
