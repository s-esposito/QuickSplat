SCANNETPP_ROOT="/home/geiger/gwb987/work/data/scannetpp"
OUTPUT_DATA_ROOT="quicksplat_spp_data_processed"

# render depth
python -m preprocess.copy_data_and_render_depth \
    --data_root $SCANNETPP_ROOT/data \
    --scene_file  quicksplat_spp_data/splits/test_scene_ids.txt \
    --output_root $OUTPUT_DATA_ROOT

# Prepare the data for training
python -m preprocess.prepare_spp_data \
    --data_root $SCANNETPP_ROOT/data \
    --scene_file  quicksplat_spp_data/splits/test_scene_ids.txt \
    --output_root $OUTPUT_DATA_ROOT \
    --voxel_size 0.02