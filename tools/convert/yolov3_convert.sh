atc --model=../model/yolov3.prototxt \
    --weight=../model/yolov3.caffemodel \
    --framework=0 \
    --output=../model/yolov3_aipp \
    --soc_version=Ascend310 \
    --insert_op_conf=../model/yolov3_aipp.cfg 
