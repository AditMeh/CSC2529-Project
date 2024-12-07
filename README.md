# CSC2529-Project

This repository borrows heavily from DINO (https://github.com/facebookresearch/dino) and minLoRA (https://github.com/cccntu/minLoRA).

## LoRA setup

1. Clone minLoRa (https://github.com/cccntu/minLoRA)
2. cd into the repo you just cloned and do `pip install -e .`

## Dataset setup:

1. Download CIFAR 10 and CIFAR 10.1 from here: https://github.com/modestyachts/CIFAR-10.1
2. Create three folders: `val`, `train` and `test`, which contain Cifar's validation set, train set and cifar 10.1 respectively

## For fine-tuning

1. Train a DINO model using the steps outlined in the original DINO repository (https://github.com/facebookresearch/dino), check the `dino` subfolder in this repository for a copy of their README with instructions on how to do this.

### For LoRA

1. Swap to the lora branch
2. Using your pretrained model, run the lora finetuning script:

```
python -m torch.distributed.launch --nproc_per_node=4 finetune_dino_lora.py --arch vit_small --patch_size 16 --warmup_teacher_temp 0.04 --teacher_temp 0.07 --warmup_teacher_temp_epochs 30 --use_fp16 false --weight_decay 0.04 --weight_decay_end 0.4 --epochs 800 --lr 0.0005 --warmup_epochs 10 --min_lr 0.00001 --global_crops_scale 0.25 1 --local_crops_scale 0.05 0.25 --local_crops_number 10 --seed 0 --optimizer adamw --momentum_teacher 0.996 --use_bn_in_head false --drop_path_rate 0.1 --data_path /voyager/datasets/oossl/CIFAR-10-images/test --output_dir ./vit_small_lora --ckpt_path ./vit_small/checkpoint.pth --lora_rank=4
```

3. Then to fit linear probes, run the linear evaluation script on your checkpoint:

```
 python -m torch.distributed.launch --nproc_per_node=4 --master_port=19231 eval_linear_lora.py --pretrained_weights vit_small_lora/checkpoint.pth --checkpoint_key teacher --data_path /voyager/datasets/oossl/CIFAR-10-images/ --eval_set test --train_set test --epochs 2 --lora_rank=4
```

### For EWC

1. Make sure you're on the main branch
2. Use your pretrained model, simply pass in `--finetune ewc` and restart from your checkpoint, the same way you trained the SSL model.


