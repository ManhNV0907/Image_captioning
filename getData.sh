# import os
mkdir coco
cd coco 
#!/bin/sh
python /content/Image_captioning/getData.py
mkdir karpathy
cd karpathy
gdown 1-o_06kAhTWfrdSGdSdkJJKbWGAyZUVHu
unzip dataset_coco.json.zip
cd ..
cd Image_captioning