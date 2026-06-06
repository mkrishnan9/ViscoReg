#!/bin/bash
# Two-phase 1D experiment:
#   Phase 1 — fit the INR to a piecewise-linear function (3valley shape)
#             using plain MSE with many random samples.
#   Phase 2 — warm-start from the Phase-1 checkpoint and train as a neural SDF
#             with the ViscoReg loss.

DIR=$(dirname $(dirname $(dirname "$(readlink -f "$0")")))
echo "If $DIR is not the correct path for your repository, set it manually at the variable DIR"
cd $DIR/sanitychecks/

BASE_LOGDIR='./log/1d_pretrain/'
mkdir -p $BASE_LOGDIR
FILE=`basename "$0"`
cp scripts/$FILE $BASE_LOGDIR

### FIXED MODEL HYPER-PARAMETERS ###
SPHERE_INIT_PARAMS=(1.6 0.1)
NEURON_TYPE='linear'
BATCH_SIZE=1
NEPOCHS=1
GRAD_CLIP_NORM=10.0
GRID_RES=256
NONMNFLD_SAMPLE_TYPE='grid'
NPOINTS=256
LOSS_WEIGHTS=(3e3 1e2 1e2 5e1 2e2)
DIVDECAY='linear'

### PHASE-1 PRETRAINING HYPER-PARAMETERS ###
PRETRAIN_LR=5e-5
PRETRAIN_STEPS=10000
PRETRAIN_BATCH=4096

### PRETRAIN SHAPE: 3valley ###
# Three valleys at -0.333, 0.0, 0.333 with zeros at the boundaries
PRETRAIN_NAME='3valley'
KX='-0.5 -0.333 -0.167 0.0 0.167 0.333 0.5'
KY='0.0 -0.167 0.0 -0.167 0.0 -0.167 0.0'

### PHASE-2 SDF HYPER-PARAMETERS ###
DECAY_PARAMS="1.0 0.3 0.5 0.6 0.1 0.8 0.01 0.0"
LOSS_TYPE='siren_wo_n_w_visc'
DIV_TYPE='full_l1'
NSAMPLES=30000
LR=5e-5
LAYERS=1
DIM=5
INIT_TYPE='siren'
NL='sine'

GPU=0  # Adjust to your available GPU

PRETRAIN_LOGDIR="${BASE_LOGDIR}pretrain/${PRETRAIN_NAME}/init_${INIT_TYPE}/l${LAYERS}_d${DIM}/"
mkdir -p $PRETRAIN_LOGDIR

echo "=== Phase 1: Pretraining on ${PRETRAIN_NAME} ==="
python3 train_pretrain_1d.py \
    --logdir $PRETRAIN_LOGDIR \
    --gpu_idx $GPU \
    --n_steps $PRETRAIN_STEPS \
    --batch_size $PRETRAIN_BATCH \
    --lr $PRETRAIN_LR \
    --decoder_n_hidden_layers $LAYERS \
    --decoder_hidden_dim $DIM \
    --nl $NL \
    --neuron_type $NEURON_TYPE \
    --init_type $INIT_TYPE \
    --sphere_init_params ${SPHERE_INIT_PARAMS[@]} \
    --keypoints_x $KX \
    --keypoints_y $KY

PRETRAIN_CKPT="${PRETRAIN_LOGDIR}trained_models/model_final.pth"
LR_TAG=$(echo "$LR" | sed 's/[e.]/_/g')
SDF_LOGDIR="${BASE_LOGDIR}sdf/${PRETRAIN_NAME}/init_${INIT_TYPE}/loss_${LOSS_TYPE}/div_${DIV_TYPE}/ns_${NSAMPLES}/lr_${LR_TAG}/l${LAYERS}_d${DIM}/"
mkdir -p $SDF_LOGDIR

SECOND_LAST=$((NSAMPLES - 2000))
EPOCHS_N_EVAL=($SECOND_LAST $NSAMPLES)

echo "=== Phase 2: SDF training with ViscoReg ==="
python3 train_basic_shape_1d.py \
    --logdir $SDF_LOGDIR --shape_type 'blob' --grid_res $GRID_RES \
    --loss_type $LOSS_TYPE --inter_loss_type 'exp' \
    --num_epochs $NEPOCHS --gpu_idx $GPU \
    --n_samples $NSAMPLES --n_points $NPOINTS \
    --batch_size $BATCH_SIZE --lr $LR \
    --nonmnfld_sample_type $NONMNFLD_SAMPLE_TYPE \
    --nonmnfld_range 0.5 \
    --decoder_n_hidden_layers $LAYERS \
    --decoder_hidden_dim $DIM \
    --div_decay $DIVDECAY --div_decay_params $DECAY_PARAMS \
    --div_type $DIV_TYPE --init_type $INIT_TYPE \
    --neuron_type $NEURON_TYPE --nl $NL \
    --sphere_init_params ${SPHERE_INIT_PARAMS[@]} \
    --loss_weights ${LOSS_WEIGHTS[@]} \
    --grad_clip_norm $GRAD_CLIP_NORM \
    --pretrain_model_path $PRETRAIN_CKPT

echo "=== Testing ==="
python3 test_basic_shape_1d.py \
    --logdir $SDF_LOGDIR --gpu_idx $GPU \
    --epoch_n "${EPOCHS_N_EVAL[@]}" \
    --neuron_type $NEURON_TYPE

echo "Done!"
