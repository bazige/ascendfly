atc --model=../modelzoo/inceptionv3.onnx \
    --framework=5 \
    --output=../modelzoo/inceptionv3_bs16 \
    --input_format=NCHW \
    --input_shape="actual_input_1:16,3,299,299" \
    --log=info \
    --soc_version=Ascend310
    