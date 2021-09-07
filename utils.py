import torch


def export_as_onnx(net, file_name, image_size=224):
    x = torch.randn(1, 3, image_size, image_size, requires_grad=True).cpu()
    torch.onnx.export(net, x, file_name, export_params=True)
