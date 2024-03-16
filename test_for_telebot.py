import os
import warnings
import matplotlib.pyplot as plt
from glob import glob
from _paths import nomeroff_net_dir

from nomeroff_net import pipeline
from nomeroff_net.tools import unzip

warnings.filterwarnings("ignore")

number_plate_detection_and_reading = pipeline("number_plate_detection_and_reading", image_loader="turbo")

current_dir = os.path.dirname(os.path.abspath(__file__))
nomeroff_net_dir = os.path.join(current_dir, "./")

image_paths = glob(os.path.join(nomeroff_net_dir, "../../datasets/ru_test/*.jpg"))
result = number_plate_detection_and_reading(image_paths, quality_profile=[3, 1, 0])

(images, images_bboxs, 
 images_points, images_zones, region_ids, 
 region_names, count_lines, 
 confidences, texts) = unzip(result)

print(texts)
