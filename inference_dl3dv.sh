DL3DV_DIR="/mnt/lustre/work/geiger/gwb929/datasets/dl3dv-colmap-sfm"
VIEWS_SPLIT="/mnt/lustre/work/geiger/gwb929/projects/optgs-unified/assets/dl3dv_evaluation/dl3dv_start_0_distance_40_ctx_8v_tgt_8v.json"

PYTHON_PATH="/home/geiger/gwb987/.conda/envs/quicksplat/bin/python"
CONFIG_PATH="configs/phase2_eval_long_no_adam.yaml"

$PYTHON_PATH inference.py \
    --config $CONFIG_PATH \
    --ckpt checkpoints/phase2.ckpt \
    --out inference_outputs \
    --one-shot \
    --skip-mesh \
    DATASET.data_format colmap \
    DATASET.image_dir images_4 \
    DATASET.views_split_path $VIEWS_SPLIT \
    DATASET.source_path $DL3DV_DIR
