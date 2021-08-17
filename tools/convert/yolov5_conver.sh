atc --model=../model/yolov5s_sim_t.onnx \
    --framework=5 \
    --output=../model/yolov5s_sim_t_aipp \
    --log=debug \
    --soc_version=Ascend310 \
    --input_shape="images:1,3,640,640" \
    --insert_op_conf=../model/yolov5s_aipp.cfg \
    # --out_nodes="Conv_277:0;Conv_261:0;Conv_245:0"
