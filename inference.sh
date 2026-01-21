TEST_DATASET_DIR="/home/geiger/gwb987/work/codebase/QuickSplat/quicksplat_spp_data_processed"

python inference.py \
    --config configs/phase2_eval.yaml \
    --ckpt checkpoints/phase2.ckpt \
    --out inference_outputs \
    DATASET.test_split_path $TEST_DATASET_DIR/splits/test_scene_ids.txt \
    DATASET.source_path $TEST_DATASET_DIR/data \
    DATASET.ply_path  $TEST_DATASET_DIR/colmap \
    DATASET.gt_ply_path $TEST_DATASET_DIR/mesh