atc --model=../model/resnet50.prototxt \
    --weight=../model/resnet50.caffemodel \
    --framework=0 \
    --output=../model/resnet50 \
    --soc_version=Ascend310 \
    --input_fp16_nodes=data \
    --output_type=FP32 \

# atc --input_shape="data:1,3,224,224" 
#     --weight="resnet50.caffemodel" 
#     --input_format=NCHW 
#     --output="resnet50" 
#     --soc_version=Ascend310 
#     --insert_op_conf=./insert_op.cfg 
#     --framework=0 
#     --model="resnet50.prototxt" 
#     --output_type=FP32