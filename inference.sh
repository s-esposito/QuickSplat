TEST_DATASET_DIR="/home/geiger/gwb987/work/codebase/QuickSplat/quicksplat_spp_data_processed"

PYTHON_PATH="~/.conda/envs/quicksplat/bin/python"
CONFIG_PATH="configs/phase2_eval_no_depth.yaml"

$PYTHON_PATH inference.py \
    --config $CONFIG_PATH \
    --ckpt checkpoints/phase2.ckpt \
    --out inference_outputs \
    DATASET.test_split_path $TEST_DATASET_DIR/splits/test_scene_ids.txt \
    DATASET.source_path $TEST_DATASET_DIR/data \
    DATASET.ply_path  $TEST_DATASET_DIR/colmap \
    DATASET.gt_ply_path $TEST_DATASET_DIR/mesh