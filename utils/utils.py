from typing import Union


# this must be solved
def check_size(
    shape: Union[int, tuple],
    stride: int
) -> Union[int, tuple]:
    if isinstance(shape, int):
        residual = shape % stride
        return residual + shape
    else:
        residual_width = shape[0] % stride
        residual_height = shape[1] % stride
        return shape[0] + residual_width, shape[1] + residual_height
