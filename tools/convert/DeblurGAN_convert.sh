atc --input_shape="blur:1,720,1280,3" \
    --input_format=NHWC \
    --output="./DeblurrGAN_pad_1280_720" \
    --soc_version=Ascend310 \
    --framework=3 \
    --model="./DeblurrGAN-pad-01051648.pb" \
    --log=info