#!/bin/bash
DIR=$(dirname $(dirname $(dirname "$(readlink -f "$0")")))
echo "If $DIR is not the correct path for your repository, set it manually at the variable DIR"
cd $DIR/sanitychecks/

BASE_LOGDIR='./log/1d/'
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
LOSS_WEIGHTS=(3e3 1e2 1e2 5e1 1e2)
DIVDECAY='linear'

### DECAY PARAMS — visc needs the full schedule, others use a simple decay ###
DECAY_VISC="1.0 0.3 0.5 0.6 0.1 0.8 0.01 0.0"
DECAY_DIV="1.0 0.5 0.0 0.0"
DECAY_NONE="0.0 0.0"

### SWEEP PARAMETERS — edit these lists to control the sweep ###
INIT_TYPES=('siren' 'geometric_relu')
LOSS_TYPES=('siren_wo_n_w_div' 'siren_wo_n_w_visc' 'siren_wo_n')
DIV_TYPES=('dir_l1' 'full_l1')
NSAMPLES_LIST=(30000)
LR_LIST=(5e-5 1e-4)
LAYERS_LIST=(1 2)
DIM_LIST=(5 2 4 8)

### AVAILABLE GPUS ###
GPUS=(0 1 2 3 4 5 6 7)
NUM_GPUS=${#GPUS[@]}
GPU_IDX=0
JOBS=()

run_and_maybe_wait() {
    JOBS+=($!)
    if [ ${#JOBS[@]} -ge $NUM_GPUS ]; then
        wait "${JOBS[@]}"
        JOBS=()
    fi
}

# ── TRAIN THEN TEST per job ────────────────────────────────────────────────────
for INIT_TYPE in "${INIT_TYPES[@]}"; do
    [ "$INIT_TYPE" == "siren" ] && NL='sine' || NL='relu'

    for LOSS_TYPE in "${LOSS_TYPES[@]}"; do
        if   [ "$LOSS_TYPE" == "siren_wo_n_w_visc" ]; then DECAY_PARAMS="$DECAY_VISC"
        elif [ "$LOSS_TYPE" == "siren_wo_n_w_div"  ]; then DECAY_PARAMS="$DECAY_DIV"
        else                                                DECAY_PARAMS="$DECAY_NONE"
        fi

        for DIV_TYPE in "${DIV_TYPES[@]}"; do
            [ "$LOSS_TYPE" == "siren_wo_n" ] && [ "$DIV_TYPE" != "${DIV_TYPES[0]}" ] && continue

            for NSAMPLES in "${NSAMPLES_LIST[@]}"; do
                SECOND_LAST=$((NSAMPLES - 2000))
                EPOCHS_N_EVAL=($SECOND_LAST $NSAMPLES)  # only last two checkpoints per sweep run

                for LR in "${LR_LIST[@]}"; do
                    for LAYERS in "${LAYERS_LIST[@]}"; do
                        for DIM in "${DIM_LIST[@]}"; do
                            GPU=${GPUS[$((GPU_IDX % NUM_GPUS))]}
                            GPU_IDX=$((GPU_IDX + 1))

                            LR_TAG=$(echo "$LR" | sed 's/[e.]/_/g')
                            LOGDIR="${BASE_LOGDIR}init_${INIT_TYPE}/loss_${LOSS_TYPE}/div_${DIV_TYPE}/ns_${NSAMPLES}/lr_${LR_TAG}/l${LAYERS}_d${DIM}/"
                            mkdir -p $LOGDIR

                            # train then immediately test in a subshell, so the
                            # GPU slot is held for both and testing starts as
                            # soon as training finishes
                            (
                                python3 train_basic_shape_1d.py \
                                    --logdir $LOGDIR --shape_type 'blob' --grid_res $GRID_RES \
                                    --loss_type $LOSS_TYPE --inter_loss_type 'exp' \
                                    --num_epochs $NEPOCHS --gpu_idx $GPU \
                                    --n_samples $NSAMPLES --n_points $NPOINTS \
                                    --batch_size $BATCH_SIZE --lr $LR \
                                    --nonmnfld_sample_type $NONMNFLD_SAMPLE_TYPE \
                                    --decoder_n_hidden_layers $LAYERS \
                                    --decoder_hidden_dim $DIM \
                                    --div_decay $DIVDECAY --div_decay_params $DECAY_PARAMS \
                                    --div_type $DIV_TYPE --init_type $INIT_TYPE \
                                    --neuron_type $NEURON_TYPE --nl $NL \
                                    --sphere_init_params ${SPHERE_INIT_PARAMS[@]} \
                                    --loss_weights ${LOSS_WEIGHTS[@]} \
                                    --grad_clip_norm $GRAD_CLIP_NORM

                                python3 test_basic_shape_1d.py \
                                    --logdir $LOGDIR --gpu_idx $GPU \
                                    --epoch_n "${EPOCHS_N_EVAL[@]}" \
                                    --neuron_type $NEURON_TYPE
                            ) &

                            run_and_maybe_wait
                            echo "Launched: init=$INIT_TYPE loss=$LOSS_TYPE div=$DIV_TYPE ns=$NSAMPLES lr=$LR l=$LAYERS d=$DIM → GPU $GPU"
                        done
                    done
                done
            done
        done
    done
done

wait
echo "All 1D experiments completed!"
