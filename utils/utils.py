import torch


def resize_long_side(
    orig_shape: tuple[int, int],
    dest_len: int,
    stride: int = 8
) -> tuple[int, int]:
    long_side_length = max(orig_shape[0], orig_shape[1])
    scale = dest_len / long_side_length

    resized_h, resized_w = round(orig_shape[0] * scale), round(orig_shape[1] * scale)
    resized_h -= resized_h % stride
    resized_w -= resized_w % stride
    return resized_h, resized_w


def scale_intrinsic_matrix(
    orig_shape: tuple[int, int],
    intrinsic: torch.Tensor,
    new_shape: tuple[int, int]
):
    intrinsic_ = intrinsic.detach().clone()
    scale_x = new_shape[1] / orig_shape[1]
    scale_y = new_shape[0] / orig_shape[0]

    intrinsic_[0][0] *= scale_x
    intrinsic_[1][1] *= scale_y
    intrinsic_[0][2] *= scale_x
    intrinsic_[1][2] *= scale_y

    return intrinsic_
