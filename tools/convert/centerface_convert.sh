# atc --model=./yolov3.prototxt \
#     --weight=./yolov3.caffemodel \
#     --framework=0 \
#     --output=./yolov3_aipp \
#     --soc_version=Ascend310 \
#     --insert_op_conf=./static_aipp_nv12.cfg

atc --model=../model/centerface.prototxt \
    --weight=../model/centerface.caffemodel \
    --framework=0 \
    --insert_op_conf=../model/aipp_centerface.config  \
    --output=../model/centerface_8batch_static_aipp_nv12 \
    --soc_version=Ascend310
