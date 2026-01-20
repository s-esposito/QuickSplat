TRAIN_DATASET_DIR=""

python inference.py \
    --config configs/phase2_eval.yaml \
    --ckpt checkpoints/phase2.ckpt \
    --out inference_outputs \
    DATASET.test_split_path $TRAIN_DATASET_DIR/splits/test_scene_ids.txt \
    DATASET.source_path $TRAIN_DATASET_DIR/data \
    DATASET.ply_path  $TRAIN_DATASET_DIR/colmap \
    DATASET.gt_ply_path $TRAIN_DATASET_DIR/mesh