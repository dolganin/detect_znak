import numpy as np
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
detect_znak_dir = os.path.join(current_dir, "../")

sys.path.append(detect_znak_dir)
CLASS_REGION_ALL = [
    "xx-unknown",
    "eu-ua-2015",
    "eu-ua-2004",
    "eu-ua-1995",
    "eu",
    "xx-transit",
    "ru",
    "kz",
    "eu-ua-ordlo-dpr",
    "eu-ua-ordlo-lpr",
    "ge",
    "by",
    "su",
    "kg",
    "am",
    "ua-military",
    "ru-military",
    "md",
    "eu-ua-custom",
]

CLASS_LINES_ALL = [
    "0",  # garbage
    "1",  # one line
    "2",  # two line
    "3",  # three line
]


# +
from detect_znak.nnmodels.numberplate_classification_model import ClassificationNet
from detect_znak.nnmodels.numberplate_options_model import NPOptionsNet
from detect_znak.pipes.number_plate_classificators.options_detector import OptionsDetector
from detect_znak.tools.mcm import get_device_torch

import torch
# np_options_net = NPOptionsNet(13, 3)
# device = get_device_torch()
# net = np_options_net.to(device)
path_to_model = '../../classifier/classifier_test.pth'
np_options_net = NPOptionsNet.load_from_checkpoint(path_to_model,
                                                       map_location=torch.device('cpu'),
                                                       region_output_size=2,
                                                       count_line_output_size=3,
                                                       img_h=100,
                                                       img_w=400,
                                                       batch_size=1,
                                                       train_regions=True,
                                                       train_count_lines=True,
                                                       backbone='mobilenet')
xs = torch.rand((1, 3, 50, 200), requires_grad=False)
# y = model(xs)
# y = OptionsDetector.unzip_predicted(y[0].detach().numpy())
# np_options_net = NPOptionsNet(13, 3)
device = 'cuda:0'
net = np_options_net
xs = torch.rand((1, 3, 50, 200))
y = np_options_net(xs)
print(y)
dynamic_axes = {'input': { 2: "inputc_h", 3: 'inputc_w'}}
torch.onnx.export(net,               # model being run
                  xs,                         # model input (or a tuple for multiple inputs)
                  "../data/models/OptionsDetector/numberplate_options/classifier_ru_kz_resnet_test.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output1'], # the model's output names
                  dynamic_axes=dynamic_axes)



