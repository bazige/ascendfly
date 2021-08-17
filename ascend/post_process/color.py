# Copyright (c) Open-MMLab. All rights reserved.
from enum import Enum

import numpy as np

from ..common.path import is_str


class Color(Enum):
    """An enum that defines common colors.

    Contains `red`, `green`, `blue`, `cyan`, `yellow`, `magenta`, `white` and `black`.
    """
    red = (0, 0, 255)
    green = (0, 255, 0)
    blue = (255, 0, 0)
    cyan = (255, 255, 0)
    yellow = (0, 255, 255)
    magenta = (255, 0, 255)
    white = (255, 255, 255)
    black = (0, 0, 0)


def color_val(color):
    """Convert various input to color tuples.

    Args:
        color (`Color`/str/tuple/int/ndarray): Color inputs

    Returns:
        tuple[int]: A tuple of 3 integers indicating BGR channels.
    """
    if is_str(color):
        return Color[color].value
    elif isinstance(color, Color):
        return color.value
    elif isinstance(color, tuple):
        assert len(color) == 3
        for channel in color:
            assert 0 <= channel <= 255
        return color
    elif isinstance(color, int):
        assert 0 <= color <= 255
        return color, color, color
    elif isinstance(color, np.ndarray):
        assert color.ndim == 1 and color.size == 3
        assert np.all((color >= 0) & (color <= 255))
        color = color.astype(np.uint8)
        return tuple(color)
    else:
        raise TypeError(f'Invalid type for color: {type(color)}')

def color_gen(class_num):
    """Convert various input to color tuples.

    Args:
        class_num (int): Class numbers

    Returns:
        list[int]: Generate colors according to class number.
    """
    assert class_num >= 0, f"Input class_num should be larger than 0."
    gen_color = []
    for _ in range(class_num):
        color = np.random.randint(0, 255, size=3).tolist()
        gen_color.append(color)
    return gen_color
