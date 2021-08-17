#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright 2020 Huawei Technologies Co., Ltd
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import ascend
from PIL import Image, ImageDraw, ImageFont
CLS = ['dog', 'cat']

class Classify(object):
    """
    Class for portrait segmentation
    """
    def __init__(self, device_id, model_path):
        if isinstance(device_id, int):
            raise TypeError(f"Input device_id expects an int, but got {type(device_id)}.")

        context = ascend.Context({device_id}).context_dict[device_id]

        self.Img = ascend.Image(context)
        self.mdl = ascend.AscendModel(context, model_path)

        (_, self.mdl_h, self.mdl_w, _) = self.mdl.tensor['data'].shape

    @utils.display_time
    def pre_process(self, image):
        """
        preprocess 
        """
        yuv_image = self.Img.imread(image)
        im_resize = self.Img.imresize(yuv_image, (self.mdl_w, self.mdl_h))

        self.mdl.feed_data({'data':im_resize})

    @utils.display_time
    def inference(self):
        """
        model inference
        """
        self.mdl.run()

    @utils.display_time
    def post_process(self, image_file, out_dir):
        """
        Post-processing, analysis of inference results
        """
        output_path = os.path.join(OUTPUT_DIR, os.path.basename(image_file))
        infer_result = self.mdl.get_tensor_by_name('').to_np
        vals = infer_result.flatten()
        pre_index = vals.argsort()[-1]
        
        img_org = Image.open(image_file)
        draw = ImageDraw.Draw(img_org)
        font = ImageFont.load_default()
        draw.text((10, 50), CLS[pre_index], font=font, fill=255)
        img_org.save(output_path)
                

def main():
    """
    main
    """
    device=0
    model_path = "../tools/model/centerface_static_aipp_nv12.om"
    image_path = './projects/shenmo/'
    out_dir    = './projects/shenmo/'

    classify = Classify(device, model_path)
    images_list = [os.path.join(image_dir, img)
                   for img in os.listdir(image_dir)
                   if os.path.splitext(img)[1] in const.IMG_EXT]

    for image_file in images_list:
        # Preprocess the picture 
        resized_image = classify.pre_process(image)

        # Inferencecd 
        result = classify.inference()

        # # Post-processing
        classify.post_process(image_file, out_dir)
         

if __name__ == '__main__':
    main()
 
