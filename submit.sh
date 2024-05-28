export CUDA_VISIBLE_DEVICES=2
cd /usr/users/vhassle/psych_track/MOTIP
python main.py --mode=submit --use-distributed=False --use-wandb=False --config-path=./configs/r50_deformable_detr_motip_dancetrack_joint_ch.yaml \
 --inference-model=./outputs/r50_deformable_detr_motip_dancetrack_joint_ch.pth --outputs-dir=./outputs/Wortschatzinsel/Neon/ --inference-dataset=Wortschatzinsel \
 --inference-split= --num-workers=0 --manual-output-dir=/usr/users/vhassle/psych_track/MOTIP/outputs/Wortschatzinsel/Neon_test_dancetrack \
 --seq-path=/usr/users/vhassle/datasets/Wortschatzinsel/Neon/2024_05_19_14_16_59.mp4 --view=no_specific_view 



