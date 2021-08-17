import sys
sys.path.append("..")
import ascend
import numpy as np
import pdb

def enc_test_001():
    stream_dir = './test_venc_h264.264'
    context = ascend.Context({1}).context_dict[1]
    Img = ascend.Image(context)

    pdb.set_trace()
    encode = ascend.Venc(context, stream_dir, 1280, 720)
    ascend.show_growth()

    for i in range(50):
        print(f"process frame:{i}")
        if i == 4:
            frame = ascend.Frame(yuv_resize, is_last=True, context=context)
        elif i%2 == 0:
            src_img = np.fromfile('./image/xiaoxin.jpg', dtype=np.uint8)
            yuv_src = Img.imdecode(src_img)
            yuv_resize = Img.imresize(yuv_src, (1280, 720))
            frame = ascend.Frame(yuv_resize)
        else:
            src_img = np.fromfile('./image/img.jpg', dtype=np.uint8)
            yuv_src = Img.imdecode(src_img)
            yuv_resize = Img.imresize(yuv_src, (1280, 720))
            frame = ascend.Frame(yuv_resize)
        encode.process(frame)

    encode.finish()
    del encode

def enc_test_002():

    stream_dir = './test_venc_h264.264'
    context = ascend.Context({0}).context_dict[0]
    Img = ascend.Image(context)

    video_stream_path = './video/cars_around_mountain_640_360.264'
    cap = ascend.VideoCapture(context, video_stream_path)

    encode = ascend.Venc(context, stream_dir, 640, 480)
    ascend.show_growth()

    while cap.is_open():
        image, frame_id = cap.read()
        if image:
            print(f"process frame:{frame_id}")
            yuv_resize = Img.imresize(image, (640, 480))
            if 0:# frame_id == 500:
                frame = ascend.Frame(yuv_resize, is_last=True, context=context)
                encode.process(frame)
                break
            else:
                frame = ascend.Frame(yuv_resize)
                encode.process(frame)

    encode.finish()
    del encode
    del cap
    del Img

enc_test_002()
