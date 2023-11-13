# Fine tuning

- Fork the original HAT repository
    - 40,846,575 parameters
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
- Stick to size of 256×256 for GT (HR) and 64x64 for LQ (LR).

## Conversion to rgb
- Normalise each image to [0, 1]
- Multiply by 255*3 so [0, 755]
- Strech across all 3 channels -> discretisation and discretisation loss is now lower
- Convert back by taking sum of 3 channels
    - Even if we have invalid values

## Commands for inference

- Navigate to `~/finetune-HAT` on Roger

```
python hat/test.py -opt options/test/HAT-L_SRx4_ImageNet-pretrain_topo.yml
```

Rewrite across multiple lines.
```
python hat/test.py \
    -opt options/test/HAT-L_SRx4_ImageNet-pretrain_topo.yml
```

## Commands for training

### Original (with my file):  
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 hat/train.py -opt options/train/train_HAT-L_SRx4_finetune_from_ImageNet_pretrain_topo.yml --launcher pytorch
```

Command line arguments:
- `-- launcher pytorch` torch.distributed
- `-m` module flag
- `-opt` flag is non-standard


### Single GPU:  
```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1  --master_port=4321 hat/train.py -opt options/train/train_HAT-L_SRx4_finetune_from_ImageNet_pretrain_topo.yml  --launcher pytorch
```

```
CUDA_VISIBLE_DEVICES=0 python \
    -m torch.distributed.launch \
    --nproc_per_node=1  \
    --master_port=4321 hat/train.py \
    --local_rank=-1 \
    -opt options/train/train_HAT-L_SRx4_finetune_from_ImageNet_pretrain_topo.yml  \
    --launcher pytorch
```

```
python \
    -m torch.distributed.launch \
    --nproc_per_node=1  hat/train.py \
    --local_rank=-1 \
    -opt options/train/train_HAT-L_SRx4_finetune_from_ImageNet_pretrain_topo.yml  \
    --launcher pytorch \
    --use-env
```


Error:
- --local_rank=-1 to disable the distributed setting
    - https://discuss.pytorch.org/t/error-unrecognized-arguments-local-rank-1/83679/5
    https://pytorch.org/docs/stable/distributed.html#launch-utility

### Torchrun now works:  
```
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc-per-node=1  hat/train.py -opt options/train/train_HAT-L_SRx4_finetune_from_ImageNet_pretrain_topo.yml
```

### Model specs
- HAT-L has 40,846,575 parameters.
    - HAT has 20.8 M parameters.
    - HAT-S has 9.6 M parameters
- One epoch (100 iterations) takes around 32 minutes on Roger.
- 50 epochs (5k iterations) takes a bit more than a day to train on Roger.
- The default for finetuning however was 2500 epochs with 250000 iterations, which is 50 x of what I am running. Thus I am only doing 2% of the default fine-tuning.  
- Number of training images: 200 (the HR images are 256 x 256 pixel (128 km x 128 km) so larger than the GP images which are 60 x 60 pixel (30 km x 30 km))
- Number of validation images: 36
- Batch size per gpu: 2 (default was 4)
- nproc per node: 
- World size (gpu number): 1
- Require iter number per epoch: 100
        Total epochs: 2500; iters: 250000.
- l_pix: pixel loss (MAE aka L1 loss)

Notes:
- Outputs are stored under experiments (non-archieved finetune) and tb_logger
- Val save_images

## ToDO:
- See demand on GPU for batch-size = 4 (20G memory for each GPU assumed)
    - almost 14 GiB at batch-size = 4
- Val frequency understand output (set to 100 now)
- Multistep LR decays the learning rate