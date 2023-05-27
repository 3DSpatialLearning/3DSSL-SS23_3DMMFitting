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
