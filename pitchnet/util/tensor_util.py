import torch.nn.functional as F


def nearest_upsample(input_tensor, target_len):
    """
    Upsample the input tensor of shape [N, C, L] to [N, C, target_len]
    using nearest neighbor interpolation method.
    """
    if input_tensor.ndim < 3:
        input_tensor = input_tensor.unsqueeze(1)
    return F.interpolate(input_tensor, target_len, mode='nearest')


def linear_upsample(input_tensor, target_len):
    """
    Upsample the input tensor of shape [N, C, L] to [N, C, target_len]
    using linear interpolation method.
    """
    if input_tensor.ndim < 3:
        input_tensor = input_tensor.unsqueeze(1)
    return F.interpolate(input_tensor, target_len, mode='linear', align_corners=True)
