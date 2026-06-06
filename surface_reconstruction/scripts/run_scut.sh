#!/bin/bash
# ViscoReg: Surface reconstruction on the SCUT real-scan dataset

DIR=$(dirname $(dirname $(dirname "$(readlink -f "$0")")))
DATASET_PATH=$DIR'/data/ScutSurf_Data/'
echo "Repository root: $DIR"
echo "SCUT dataset path: $DATASET_PATH"

cd $DIR/surface_reconstruction/

LOGDIR='./log/scut/'
mkdir -p $LOGDIR
cp scripts/run_scut.sh $LOGDIR
SCAN_PATH=$DATASET_PATH'/real_object_scan/'

### MODEL HYPER-PARAMETERS ###
LAYERS=4
DECODER_HIDDEN_DIM=256
NL='sine'
SPHERE_INIT_PARAMS=(1.6 0.1)
INIT_TYPE='siren'
NEURON_TYPE='linear'

### LOSS HYPER-PARAMETERS ###
LOSS_TYPE='siren_wo_n_w_visc'
LOSS_WEIGHTS=(3e3 1e2 1e2 5e1 1.0)
DIV_TYPE='dir_l1'
DIV_DECAY='linear'
DECAY_PARAMS=(1.0 0.6 0.0 0.0)

### DOMAIN HYPER-PARAMETERS ###
GRID_RES=256
TEST_GRID_RES=512
NONMNFLD_SAMPLE_TYPE='grid'
NPOINTS=15000

### TRAINING HYPER-PARAMETERS ###
NITERATIONS=20000
LR=1e-4
GRAD_CLIP_NORM=10.0

GPUS=(0 1 2 3 4 5 6 7)  # Adjust to your available GPUs

FILENAMES=(
    'bottle_shampoo_pcd.ply'
    'bowl_chinese_pcd.ply'
    'cloth_duck_pcd.ply'
    'coffe_bottle_metal_pcd.ply'
    'coffe_bottle_plastic_pcd.ply'
    'cup1_pcd.ply'
    'flower_pot_2_pcd.ply'
    'flower_pot_pcd.ply'
    'gift_box_pcd.ply'
    'lock_pengfen_pcd.ply'
    'marker_pcd.ply'
    'mouse_two_pcd.ply'
    'rabbit_pcd.ply'
    'romoter_pcd.ply'
    'screwnew_pcd.ply'
    'tap2_pcd.ply'
    'toy_cat_pcd.ply'
    'toy_duck_pcd.ply'
    'wrench_pcd.ply'
    'xiaojiejie2_pcd.ply'
)

IDENTIFIER='viscoreg'
NUM_GPUS=${#GPUS[@]}

mkdir -p "$LOGDIR$IDENTIFIER"

NUM_FILES=${#FILENAMES[@]}
for ((i=0; i<$NUM_FILES; i+=NUM_GPUS)); do
    PIDS=()
    for ((j=0; j<NUM_GPUS; j++)); do
        IDX=$((i + j))
        [ $IDX -ge $NUM_FILES ] && break
        FILENAME="${FILENAMES[$IDX]}"
        GPU=${GPUS[$((j % NUM_GPUS))]}
        CUDA_VISIBLE_DEVICES=$GPU python3 train_surface_reconstruction.py \
            --logdir $LOGDIR$IDENTIFIER/real_object_scan \
            --file_name $FILENAME \
            --grid_res $GRID_RES \
            --loss_type $LOSS_TYPE \
            --gpu_idx 0 \
            --n_iterations $NITERATIONS \
            --n_points $NPOINTS \
            --lr ${LR} \
            --nonmnfld_sample_type $NONMNFLD_SAMPLE_TYPE \
            --dataset_path $SCAN_PATH \
            --decoder_n_hidden_layers $LAYERS \
            --decoder_hidden_dim ${DECODER_HIDDEN_DIM} \
            --div_decay $DIV_DECAY \
            --div_decay_params ${DECAY_PARAMS[@]} \
            --div_type $DIV_TYPE \
            --init_type ${INIT_TYPE} \
            --neuron_type ${NEURON_TYPE} \
            --nl ${NL} \
            --sphere_init_params ${SPHERE_INIT_PARAMS[@]} \
            --loss_weights ${LOSS_WEIGHTS[@]} \
            --grad_clip_norm ${GRAD_CLIP_NORM} &
        PIDS+=($!)
    done
    wait "${PIDS[@]}"

    PIDS=()
    for ((j=0; j<NUM_GPUS; j++)); do
        IDX=$((i + j))
        [ $IDX -ge $NUM_FILES ] && break
        FILENAME="${FILENAMES[$IDX]}"
        GPU=${GPUS[$((j % NUM_GPUS))]}
        CUDA_VISIBLE_DEVICES=$GPU python3 test_surface_reconstruction.py \
            --logdir $LOGDIR$IDENTIFIER/real_object_scan \
            --file_name $FILENAME \
            --export_mesh 1 \
            --dataset_path $SCAN_PATH \
            --grid_res $TEST_GRID_RES \
            --gpu_idx 0 &
        PIDS+=($!)
    done
    wait "${PIDS[@]}"
done

python3 compute_metrics_scut.py \
    --results_path $LOGDIR$IDENTIFIER/real_object_scan/result_meshes \
    --dataset_path $DATASET_PATH \
    --div_type $DIV_TYPE \
    --neuron_type ${NEURON_TYPE} \
    --decoder_n_hidden_layers $LAYERS \
    --decoder_hidden_dim ${DECODER_HIDDEN_DIM} \
    --nl ${NL}
