atc --model=faster_rcnn.prototxt \
    --weight=faster_rcnn.caffemodel \
    --framework=0 \
    --output=faster_rcnn_caffe_ascend310_aipp \
    --soc_version=Ascend310 \
    --input_shape="data:1,3,-1,-1;im_info:1,3" \
    --dynamic_image_size="512,512;600,600;800,800"