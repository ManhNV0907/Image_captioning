# Image_captioning
This is a PyTorch implementation of the Image Captioning model using Transformer based model.
## Dataset
* MSCOCO '14 Dataset
* Andrej Karpathy's split dataset 

To download the dataset, run this command:
```
!bash getData.sh
```
## Refactor the root as follow:
```
root/
├── coco/
│   ├── annotations/
│   │   ├── captions_train2014.json
│   │   └── captions_train2014.json
│   ├── karpathy/
│   │   └── dataset_coco.json
│   ├── train2014/
│   └── val2014/
│
└── Image_captioning/ 
    ├── images/
    ├── pretrained/
    ├── results/
    ├── caption.py
    ├── datasets.py
    ├── eval.py
    ├── getData.py
    ├── getData.sh
    ├── models.py
    ├── README.md
    ├── train.py
    └── utils.py
```
## Perform Training
```
python train.py \
    --embedding_dim 512 \
    --tokenizer bert-base-uncased \
    --max_seq_len 128 \
    --encoder_layers 6 \
    --decoder_layers 12 \
    --num_heads 8 \
    --dropout 0.1 \
    --model_path ./pretrained/model_image_captioning_eff_transfomer.pt \
    --device cuda:0 \
    --batch_size 24 \
    --n_epochs 25 \
    --learning_rate 1e-4 \
    --early_stopping 5 \
    --image_dir ../coco/ \
    --karpathy_json_path ../coco/karpathy/dataset_coco.json \
    --val_annotation_path ../coco/annotations/captions_val2014.json \
    --log_path ./images/log_training.json \
    --log_visualize_dir ./images/

```
## Perform Evaluation
```
python evaluation.py \
    --embedding_dim 512 \
    --tokenizer bert-base-uncased \
    --max_seq_len 128 \
    --encoder_layers 6 \
    --decoder_layers 12 \
    --num_heads 8 \
    --dropout 0.1 \
    --model_path ./pretrained/model_image_captioning_eff_transfomer.pt \
    --device cuda:0 \
    --image_dir ../coco/ \
    --karpathy_json_path ../coco/karpathy/dataset_coco.json \
    --val_annotation_path ../coco/annotations/captions_val2014.json \
    --output_dir ./results/
```
## Inference 
 If you don't have resouces for training, you can download the pretrained model from [here](https://drive.google.com/file/d/1Qdo8ab9Ux-6jH5RMn4CBLHGYfQMuWQNb/view). Then put the dowloaded file in pretrained folder.
 ```
 python caption.py \
    --embedding_dim 512 \
    --tokenizer bert-base-uncased \
    --max_seq_len 128 \
    --encoder_layers 6 \
    --decoder_layers 12 \
    --num_heads 8 \
    --dropout 0.1 \
    --model_path ./pretrained/model_image_captioning_eff_transfomer.pt \
    --device cuda:0 \
    --beam_size 3 
 ```

