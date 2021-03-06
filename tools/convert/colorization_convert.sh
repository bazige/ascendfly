atc --output_type=FP32 \
    --input_shape="data_l:1,1,224,224" \
    --model="../model/colorization.prototxt" \
    --weight="../model/colorization.caffemodel" \
    --input_format=NCHW \
    --output="../model/colorization" \
    --soc_version=Ascend310 \
    --framework=0 \
    --save_original_model=false
