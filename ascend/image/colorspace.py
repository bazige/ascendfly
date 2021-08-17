# Copyright (c) Open-MMLab. All rights reserved.

import numpy as np
from ..data.ascendarray import AscendArray
from ..ops.blas import Matmul, Vmul


def _convert_input_type_range(img):
    """Convert the type and range of the input image.

    It converts the input image to np.float16 type and range of [0, 1].
    It is mainly used for pre-processing the input image in colorspace
    convertion functions such as rgb2ycbcr and ycbcr2rgb.

    Args:
        img (AscendArray): The input image. It accepts:
        1. np.uint8 type with range [0, 255];
        2. np.float16 type with range [0, 1].

    Returns:
        (AscendArray): The converted image with type of np.float16 and range of [0, 1].
    """
    img_type = img.dtype
    if img_type == np.float16:
        pass
    elif img_type == np.uint8:
        img = img.astype(np.float16)
        '''
        to do normalize
        '''
        img /= 255.
    else:
        raise TypeError('The img type should be np.float16 or np.uint8, '
                        f'but got {img_type}')
    return img


def _convert_output_type_range(img, dst_type):
    """Convert the type and range of the image according to dst_type.

    It converts the image to desired type and range. If `dst_type` is np.uint8,
    images will be converted to np.uint8 type with range [0, 255]. If
    `dst_type` is np.float32/np.float16, it converts the image to np.float32/
    np.float32 type with range [0, 1].
    It is mainly used for post-processing images in colorspace convertion
    functions such as rgb2ycbcr and ycbcr2rgb.

    Args:
        img (AscendArray): The image to be converted with np.float32 type and
            range [0, 255].
        dst_type (np.uint8 | np.float32): If dst_type is np.uint8, it
            converts the image to np.uint8 type with range [0, 255]. If
            dst_type is np.float32, it converts the image to np.float32 type
            with range [0, 1].

    Returns:
        (AscendArray): The converted image with desired type and range.
    """
    if dst_type not in (np.uint8, np.float16, np.float32):
        raise TypeError('The dst_type should be np.float32, np.float16 or np.uint8, '
                        f'but got {dst_type}')
    if dst_type == np.uint8:
        # img = img.round()
        pass
    else:
        '''
        to do normalize
        '''
        img /= 255.
    return img.astype(dst_type)


def rgb2ycbcr(img, y_only=False):
    """Convert a RGB image to YCbCr image.

    This function produces the same results as Matlab's `rgb2ycbcr` function.
    It implements the ITU-R BT.601 conversion for standard-definition
    television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.

    It differs from a similar function in cv2.cvtColor: `RGB <-> YCrCb`.
    In OpenCV, it implements a JPEG conversion. See more details in
    https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.

    Args:
        img (AscendArray): The input image. It accepts:

        - np.uint8 type with range [0, 255];
        - np.float16 type with range [0, 1].

        y_only (bool): Whether to only return Y channel. Default: False.

    Returns:
        AscendArray: The converted YCbCr image. The output image has the same type
            and range as input image.
    """
    img_type = img.dtype
    img = _convert_input_type_range(img)
    if y_only:
        # out_img = np.dot(img, [65.481, 128.553, 24.966]) + 16.0
        trans = np.array([65.481, 128.553, 24.966], dtype=np.float16)
        bias = np.array([16.0], dtype=np.float16)
    else:
        trans = np.array([[ 65.481, -37.797,   112.0], 
                          [128.553, -74.203, -93.786],
                          [ 24.966,   112.0, -18.214]], dtype=np.float16)
        bias = np.array([16, 128, 128], dtype=np.float16)

    # clone to device
    trans = AscendArray.clone(trans)
    bias = AscendArray.clone(bias)

    if y_only:
        # do transmit
        out_inst = Vmul(img, trans, bias)
    else:
        # do transmit
        out_inst = Matmul(img, trans, bias)
    out_img = _convert_output_type_range(out_inst.out, img_type)
    return out_img


def bgr2ycbcr(img, y_only=False):
    """Convert a BGR image to YCbCr image.

    The bgr version of rgb2ycbcr.
    It implements the ITU-R BT.601 conversion for standard-definition
    television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.

    It differs from a similar function in cv2.cvtColor: `BGR <-> YCrCb`.
    In OpenCV, it implements a JPEG conversion. See more details in
    https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.

    Args:
        img (ndarray): The input image. It accepts:

        - np.uint8 type with range [0, 255].
        - np.float16 type with range [0, 1].

        y_only (bool): Whether to only return Y channel. Default: False.

    Returns:
        ndarray: The converted YCbCr image. The output image has the same type
            and range as input image.
    """
    img_type = img.dtype
    img = _convert_input_type_range(img)
    if y_only:
        # out_img = np.dot(img, [24.966, 128.553, 65.481]) + 16.0
        trans = np.array([24.966, 128.553, 65.481], dtype=np.float16)
        bias = np.array([16.0], dtype=np.float16)
    else:
        trans = np.array([[ 24.966,   112.0, -18.214], 
                          [128.553, -74.203, -93.786],
                          [ 65.481, -37.797,   112.0]], dtype=np.float16)
        bias = np.array([16, 128, 128], dtype=np.float16)

    # clone to device
    trans = AscendArray.clone(trans)
    bias = AscendArray.clone(bias)

    if y_only:
        out_inst = Vmul(img, trans, bias)
    else:
         # do transmit
        out_inst = Matmul(img, trans, bias)
    out_img = _convert_output_type_range(out_inst.out, img_type)
    return out_img


def ycbcr2rgb(img):
    """Convert a YCbCr image to RGB image.

    This function produces the same results as Matlab's ycbcr2rgb function.
    It implements the ITU-R BT.601 conversion for standard-definition
    television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.

    It differs from a similar function in cv2.cvtColor: `YCrCb <-> RGB`.
    In OpenCV, it implements a JPEG conversion. See more details in
    https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.

    Args:
        img (ndarray): The input image. It accepts:
        
        - np.uint8 type with range [0, 255];
        - np.float16 type with range [0, 1].

    Returns:
        ndarray: The converted RGB image. The output image has the same type
            and range as input image.
    """
    img_type = img.dtype
    '''
    to do denormalize
    '''
    img = _convert_input_type_range(img) * 255

    trans = np.array([[0.00456621,  0.00456621, 0.00456621], 
                      [         0, -0.00153632, 0.00791071],
                      [0.00625893, -0.00318811,          0]], dtype=np.float16)

    bias = np.array([-222.921, 135.576, -276.836], dtype=np.float16) 

    # clone to device
    trans = AscendArray.clone(trans)
    bias = AscendArray.clone(bias)

    # do transmit
    out_inst = Matmul(img, trans, bias, alpha=255.0)
    out_img = _convert_output_type_range(out_inst.out, img_type)
    return out_img


def ycbcr2bgr(img):
    """Convert a YCbCr image to BGR image.

    It implements the ITU-R BT.601 conversion for standard-definition
    television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.

    It differs from a similar function in cv2.cvtColor: `YCrCb <-> BGR`.
    In OpenCV, it implements a JPEG conversion. See more details in
    https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.

    Args:
        img (ndarray): The input image. It accepts:

        - np.uint8 type with range [0, 255];
        - np.float16 type with range [0, 1].

    Returns:
        ndarray: The converted BGR image. The output image has the same type
            and range as input image.
    """
    img_type = img.dtype
    '''
    to do denormalize
    '''
    img = _convert_input_type_range(img) * 255

    trans = np.array([[0.00456621,  0.00456621, 0.00456621], 
                      [0.00791071, -0.00153632,          0],
                      [         0, -0.00318811, 0.00625893]], dtype=np.float16)

    bias = np.array([-276.836, 135.576, -222.921], dtype=np.float16) 

    # clone to device
    trans = AscendArray.clone(trans)
    bias = AscendArray.clone(bias)

    # do transmit
    out_inst = Matmul(img, trans, bias, alpha=255.0)

    out_img = _convert_output_type_range(out_inst.out, img_type)
    return out_img

def hsv2bgr(img):
    """Convert a HSV image to BGR image. It released by hsv2bgr operator.

    Args:
        img (AscendArray): The input image. It accepts:

        - np.uint8 type with range [0, 255];
        - np.float16 type with range [0, 1].

    Returns:
        AscendArray: The converted BGR image. The output image has the same type
            and range as input image.
    """
    pass

def bgr2hsv(img):
    """Convert a BGR image to HSV image. It released by bgr2hsv operator.

    Args:
        img (AscendArray): The input image. It accepts:

        - np.uint8 type with range [0, 255];
        - np.float16 type with range [0, 1].

    Returns:
        AscendArray: The converted HSV image. The output image has the same type
            and range as input image.
    """
    pass
