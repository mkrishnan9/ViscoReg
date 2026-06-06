#!/bin/bash
# ViscoReg: Surface reconstruction on the Surface Reconstruction Benchmark (SRB)

DIR=$(dirname $(dirname $(dirname "$(readlink -f "$0")")))
DATASET_PATH=$DIR'/data/deep_geometric_prior_data'
echo "Repository root: $DIR"
echo "SRB dataset path: $DATASET_PATH"

cd $DIR/surface_reconstruction/

LOGDIR='./log/srb/'
mkdir -p $LOGDIR
cp scripts/run_srb.sh $LOGDIR
SCAN_PATH=$DATASET_PATH'/scans/'

### MODEL HYPER-PARAMETERS ###
LAYERS=5
DECODER_HIDDEN_DIM=128
NL='sine'
SPHERE_INIT_PARAMS=(1.6 0.1)
INIT_TYPE='mfgi'
NEURON_TYPE='linear'

### LOSS HYPER-PARAMETERS ###
LOSS_TYPE='siren_wo_n_w_visc'
LOSS_WEIGHTS=(3e3 1e2 1e2 5e1 0.5)
DIV_TYPE='dir_l1'
DIV_DECAY='linear'
DECAY_PARAMS=(0.5 0.2 0.4 0.4 0.04 0.6 0.005 0.8 0.0 0.0)

### DOMAIN HYPER-PARAMETERS ###
GRID_RES=256
TEST_GRID_RES=512
NONMNFLD_SAMPLE_TYPE='grid'
NPOINTS=15000

### TRAINING HYPER-PARAMETERS ###
NITERATIONS=10000
LR=1e-4
GRAD_CLIP_NORM=10.0

GPUS=(0 1 2 3 4)  # Adjust to your available GPUs
FILENAMES=('gargoyle.ply' 'daratech.ply' 'lord_quas.ply' 'anchor.ply' 'dc.ply')

IDENTIFIER='viscoreg'

mkdir -p "$LOGDIR$IDENTIFIER"

PIDS=()
for i in "${!FILENAMES[@]}"; do
    FILENAME="${FILENAMES[i]}"
    GPU=${GPUS[$((i % ${#GPUS[@]}))]}
    CUDA_VISIBLE_DEVICES=$GPU python3 train_surface_reconstruction.py \
        --logdir $LOGDIR$IDENTIFIER \
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
for i in "${!FILENAMES[@]}"; do
    FILENAME="${FILENAMES[i]}"
    GPU=${GPUS[$((i % ${#GPUS[@]}))]}
    CUDA_VISIBLE_DEVICES=$GPU python3 test_surface_reconstruction.py \
        --logdir $LOGDIR$IDENTIFIER \
        --file_name $FILENAME \
        --export_mesh 1 \
        --dataset_path $SCAN_PATH \
        --grid_res $TEST_GRID_RES \
        --gpu_idx 0 &
    PIDS+=($!)
done
wait "${PIDS[@]}"

RESULTS_DIR=$LOGDIR$IDENTIFIER'/result_meshes'
python3 compute_metrics_srb.py \
    --logdir $LOGDIR$IDENTIFIER \
    --dataset_path $DATASET_PATH \
    --results_path $RESULTS_DIR
