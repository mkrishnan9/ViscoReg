#!/bin/bash
# ViscoReg: Scene reconstruction (interior room from SIREN paper)

DIR=$(dirname $(dirname $(dirname "$(readlink -f "$0")")))
DATASET_PATH=$DIR'/data/scene_reconstruction/'
echo "Repository root: $DIR"
echo "Scene dataset path: $DATASET_PATH"

cd $DIR/surface_reconstruction/

LOGDIR='./log/scene/'
mkdir -p $LOGDIR
cp scripts/run_scene.sh $LOGDIR
FILENAME='interior_room.ply'

### MODEL HYPER-PARAMETERS ###
LAYERS=8
DECODER_HIDDEN_DIM=512
NL='sine'
SPHERE_INIT_PARAMS=(1.6 0.1)
INIT_TYPE='siren'
NEURON_TYPE='linear'

### LOSS HYPER-PARAMETERS ###
LOSS_TYPE='siren_wo_n_w_visc'
LOSS_WEIGHTS=(5e3 1e2 1e2 5e1 0.5)
DIV_TYPE='dir_l1'
DIV_DECAY='linear'
DECAY_PARAMS=(0.5 0.5 0.01 0.6 0.0 0.0)

### DOMAIN HYPER-PARAMETERS ###
GRID_RES=256
TEST_GRID_RES=512
NONMNFLD_SAMPLE_TYPE='grid'
NPOINTS=15000

### TRAINING HYPER-PARAMETERS ###
NITERATIONS=100000
GPU=0
LR=8e-6
GRAD_CLIP_NORM=10.0

IDENTIFIER='viscoreg'

mkdir -p $LOGDIR$IDENTIFIER

python3 train_surface_reconstruction.py \
    --logdir $LOGDIR$IDENTIFIER \
    --file_name $FILENAME \
    --grid_res $GRID_RES \
    --loss_type $LOSS_TYPE \
    --gpu_idx $GPU \
    --n_iterations $NITERATIONS \
    --n_points $NPOINTS \
    --lr $LR \
    --nonmnfld_sample_type ${NONMNFLD_SAMPLE_TYPE} \
    --dataset_path $DATASET_PATH \
    --decoder_n_hidden_layers $LAYERS \
    --decoder_hidden_dim $DECODER_HIDDEN_DIM \
    --div_decay $DIV_DECAY \
    --div_decay_params ${DECAY_PARAMS[@]} \
    --div_type ${DIV_TYPE} \
    --init_type ${INIT_TYPE} \
    --neuron_type ${NEURON_TYPE} \
    --nl ${NL} \
    --sphere_init_params ${SPHERE_INIT_PARAMS[@]} \
    --loss_weights ${LOSS_WEIGHTS[@]} \
    --grad_clip_norm ${GRAD_CLIP_NORM}

python3 test_surface_reconstruction.py \
    --logdir $LOGDIR$IDENTIFIER \
    --file_name $FILENAME \
    --export_mesh 1 \
    --dataset_path $DATASET_PATH \
    --grid_res $TEST_GRID_RES \
    --gpu_idx $GPU
