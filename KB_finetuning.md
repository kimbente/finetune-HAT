# Fine tuning

- Fork the original HAT repository
- Made a new .conda environment
- Adapt .yml file for finetuning
    - Input: `options/train/train_HAT-L_SRx4_finetune_from_ImageNet_pretrain.yml`
    - Own: `options/train/train_HAT-L_SRx4_finetune_from_ImageNet_pretrain_topo.yml`
    - Training data and validation data
- Training and validation dataset preperation: [Instructions](https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md) 
    - [PairedImageDataset class](https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/data/paired_image_dataset.py)
    - Training paths: 
        - `.datasets/ANT_train/ANT_train_HR_sub`
        - `.datasets/ANT_train/ANT_train_LR_sub/X4_sub`
        - LR image: normalise: strtch across 3 channels leaving 25 and 25 on either
        - needs to be reversible https://data.vision.ee.ethz.ch/cvl/DIV2K/
- Download large HAT-L pretrained model (.pth file) from Google drive using `gdown https://drive.google.com/uc?id=1uefIctjoNE3Tg6GTzelesTTshVogQdUf` (note: Google drive link was adjusted with uc?id= to fit the right format see [Stackoverflow post](https://stackoverflow.com/questions/25010369/wget-curl-large-file-from-google-drive))
    - Relative path: `./experiments/pretrained_models/HAT-L_SRx4_ImageNet-pretrain.pth`
- Command: 
    - make sure I am in ~/finetune-HAT
    - CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 hat/train.py -opt options/train/train_HAT_SRx2_from_scratch.yml --launcher pytorch
    - CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 hat/train.py -opt options/train/train_HAT-L_SRx4_finetune_from_ImageNet_pretrain_topo.yml --launcher pytorch

hat/train.py