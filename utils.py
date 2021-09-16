import os

import torch
from torch.onnx import *
from ofa.utils import *


def export_as_onnx(net, file_name, image_size=224):
    x = torch.randn(1, 3, image_size, image_size, requires_grad=True).cpu()
    torch.onnx.export(net, x, file_name, export_params=True)


def export_as_dynamic_onnx(net, file_name, image_size=224):
    x = torch.randn(1, 3, image_size, image_size, requires_grad=True).cpu()
    torch.onnx.export(net, x, file_name, export_params=True, operator_export_type=OperatorExportTypes.ONNX_ATEN_FALLBACK)


def count_flops(net, input_shape=(3, 32, 32)):
    return {'flops': count_net_flops(net, [1] + list(input_shape)) / 1e6}


def project_root():
    return os.path.dirname(os.path.abspath(__file__))
