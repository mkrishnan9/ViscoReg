#!/bin/bash
# ViscoReg: Surface reconstruction on ShapeNet (NSP split)
# Trains and evaluates ViscoReg on all 13 ShapeNet categories in parallel.

DIR=$(dirname $(dirname $(dirname "$(readlink -f "$0")")))
DATASET_PATH=$DIR'/data/NSP_dataset/'
RAW_DATASET_PATH=$DIR'/data/ShapeNetNSP/'
echo "Repository root: $DIR"
echo "NSP dataset path: $DATASET_PATH"

cd $DIR/surface_reconstruction/

LOGDIR='./log/shapenet/'
mkdir -p $LOGDIR
cp scripts/run_shapenet.sh $LOGDIR

### MODEL HYPER-PARAMETERS ###
LAYERS=4
DECODER_HIDDEN_DIM=256
NL='sine'
SPHERE_INIT_PARAMS=(1.6 0.1)
INIT_TYPE='mfgi'
NEURON_TYPE='linear'

### LOSS HYPER-PARAMETERS ###
LOSS_TYPE='siren_wo_n_w_visc'
LOSS_WEIGHTS=(3e3 1e2 1e2 5e1 1.0)
DIV_TYPE='dir_l1'
DIV_DECAY='linear'
DECAY_PARAMS=(1.0 0.3 0.25 0.4 0.0 0.0)

### DOMAIN HYPER-PARAMETERS ###
GRID_RES=256
TEST_GRID_RES=512
NONMNFLD_SAMPLE_TYPE='grid'
NPOINTS=15000

### TRAINING HYPER-PARAMETERS ###
NITERATIONS=10000
LR=5e-5
GRAD_CLIP_NORM=10.0

GPUS=(0 1 2 3 4)  # Adjust to your available GPUs

IDENTIFIER='viscoreg'

for FOLDER_PATH in ${DATASET_PATH}/*/; do
    FOLDER_NAME="$(basename "$FOLDER_PATH")"
    FOLDER_PATH=${DATASET_PATH}/$FOLDER_NAME/
    echo "=== Category: $FOLDER_NAME ==="

    FILES=($FOLDER_PATH/*)
    NUM_FILES=${#FILES[@]}
    NUM_GPUS=${#GPUS[@]}

    for ((i=0; i<$NUM_FILES; i+=NUM_GPUS)); do
        PIDS=()
        for ((j=0; j<NUM_GPUS; j++)); do
            IDX=$((i + j))
            [ $IDX -ge $NUM_FILES ] && break
            FILENAME="$(basename "${FILES[$IDX]}")"
            GPU=${GPUS[$((j % NUM_GPUS))]}
            CUDA_VISIBLE_DEVICES=$GPU python3 train_surface_reconstruction.py \
                --logdir $LOGDIR/$IDENTIFIER/$FOLDER_NAME \
                --file_name $FILENAME \
                --grid_res $GRID_RES \
                --loss_type $LOSS_TYPE \
                --gpu_idx 0 \
                --n_iterations $NITERATIONS \
                --n_points $NPOINTS \
                --lr ${LR} \
                --nonmnfld_sample_type $NONMNFLD_SAMPLE_TYPE \
                --dataset_path $FOLDER_PATH \
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
            FILENAME="$(basename "${FILES[$IDX]}")"
            GPU=${GPUS[$((j % NUM_GPUS))]}
            CUDA_VISIBLE_DEVICES=$GPU python3 test_surface_reconstruction.py \
                --logdir $LOGDIR/$IDENTIFIER/$FOLDER_NAME \
                --file_name $FILENAME \
                --export_mesh 1 \
                --dataset_path $FOLDER_PATH \
                --grid_res $TEST_GRID_RES \
                --gpu_idx 0 &
            PIDS+=($!)
        done
        wait "${PIDS[@]}"
    done
done

python3 compute_metrics_shapenet.py \
    --logdir $LOGDIR$IDENTIFIER \
    --dataset_path $DATASET_PATH \
    --raw_dataset_path $RAW_DATASET_PATH \
    --div_type $DIV_TYPE \
    --neuron_type ${NEURON_TYPE} \
    --decoder_n_hidden_layers $LAYERS \
    --decoder_hidden_dim ${DECODER_HIDDEN_DIM} \
    --nl ${NL}
