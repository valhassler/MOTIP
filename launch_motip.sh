#!/bin/bash

# Set environment variables
export CUDA_VISIBLE_DEVICES="3"
# Define the working directory
#cd /usr/users/vhassle/psych_track/MOTIP

# Execute the Python program with the specified arguments
python /usr/users/vhassle/psych_track/MOTIP/main.py \
    --mode=submit \
    --use-distributed=False \
    --use-wandb=False \
    --config-path=./configs/r50_deformable_detr_motip_test.yaml \
    --inference-model=./outputs/r50_deformable_detr_motip_mot17.pth \
    --outputs-dir=./outputs/Wortschatzinsel/Neon/ \
    --inference-dataset=Wortschatzinsel \
    --inference-split= \
    --num-workers=0 \
    --manual-output-dir=/usr/users/vhassle/psych_track/MOTIP/outputs/Wortschatzinsel/Neon_test \
    --seq-path=/usr/users/vhassle/datasets/Wortschatzinsel/Neon/2024_05_19_14_16_59.mp4 \
    --view=top \
    --det-thresh=0.4
    # --view=no_specific_view \

