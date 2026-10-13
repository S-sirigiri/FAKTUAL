#!/bin/bash
set -e


DEBUG=1
SLURM_HOSTNAME="<enter_hostname>"
SLURM_SBATCH_FILE="<enter_sbatch_file>"

INT_ID="${1:-}"
CMD_INDEX=0    # how many commands we've seen (via run_cmd)
FOUND_NTH=0    # set to 1 once we print the N-th command

function run_cmd {
    # If user asked for the N-th command, count and only print that one (no exec).
    if [[ -n "${INT_ID}" ]]; then
        ((++CMD_INDEX))
        if [[ "${CMD_INDEX}" -eq "${INT_ID}" ]]; then
            echo "${CMD}"     # return the N-th command
            FOUND_NTH=1
            exit 0           # stop after returning it
        else
            return 0         # skip execution/printing for non-matching commands
        fi
    fi

    # Normal behavior (no INT_ID provided)
    echo ""
    echo "${CMD}"
    if [[ ${DEBUG} == 0 ]]; then
        if [[ "$(hostname)" == "${SLURM_HOSTNAME}" ]]; then
            sbatch "${SLURM_SBATCH_FILE}" "${CMD}"
        else
            eval "${CMD}"
        fi
    fi
}


function get_test_start_seed {
    if [[ "${ckpt}" =~ ^[0-9]+$ ]]; then
        run_test_start_seed=$((test_start_seed + (seed * num_episodes) + (ckpt + 1) * 100000))
    elif [[ "${ckpt}" == "latest" || "$ckpt" == "best" ]]; then
        run_test_start_seed=$((test_start_seed + (seed * num_episodes)))
    fi
}


function eval_save_episodes {
    # Training directory (policy checkpoint).
    train_name="${train_date}_train_${policy}_${task}_${seed}"

    # Additional commands.
    get_test_start_seed

    # Setup.
    CMD="python ${script}.py"
    CMD="${CMD} --output_dir=${output_dir}/${date}/${train_name}/${ckpt}"
    CMD="${CMD} --train_dir=${train_output_dir}/${train_date}/${train_name}"
    CMD="${CMD} --train_ckpt=${ckpt}"
    CMD="${CMD} --num_episodes=${num_episodes}"
    CMD="${CMD} --test_start_seed=${run_test_start_seed}"
    CMD="${CMD} --overwrite=${overwrite}"
    CMD="${CMD} --device=${device}"
    run_cmd
}


function eval_save_episodes_over_ckpts {
    ckpt="${train_ckpt}"
    if [[ ! "${ckpt}" =~ [[:space:]] ]]; then
        eval_save_episodes
    else
        IFS=' ' read -r -a ckpts <<< "${ckpt}"
        for ckpt in "${ckpts[@]}"; do
            eval_save_episodes
        done
    fi
}


function eval_save_episodes_over_seeds {
    for task in "${TASKS[@]}"; do
        for method in "${METHODS[@]}"; do
            policy="${POLICIES[${method}_${state}]}"
            
            # Compute over seeds.
            seed="${train_seed}"
            if [[ "${seed}" =~ ^[0-9]+$ ]]; then
                eval_save_episodes_over_ckpts
            else
                IFS=' ' read -r -a train_seeds <<< "${seed}"
                for seed in "${train_seeds[@]}"; do
                    eval_save_episodes_over_ckpts
                done
            fi
        done
    done
}


# Tasks.
TASKS=(
    ## Official tasks.
    "lift_mh"
    "square_mh"
    "transport_mh"

    ## Supported tasks.
    "can_mh"
    # "can_ph"
    # "lift_ph"
    # "square_ph"
    # "transport_ph"
    # "tool_hang_ph"
    # "pusht"
    
    ## Unsupported tasks.
    # "block_pushing"
    # "kitchen"
)

# Methods.
METHODS=(
    "diffusion_policy_cnn"
)

# Policies.
declare -A POLICIES=(
    # low_dim.
    ["diffusion_policy_cnn_low_dim"]="diffusion_unet_lowdim"

    # image.
    ["diffusion_policy_cnn_image"]="diffusion_unet_image"
)



######################## Experiment setup. ########################
script="eval_save_episodes"
output_dir="data/outputs/${script}"
train_output_dir="data/outputs/train"
project="cupid"
device="cuda:0"

# General experiment params.
overwrite=0
test_start_seed=100000
num_episodes=100

# Policies.
train_seed="0 1 2"
train_demoscore=0
if [[ $train_demoscore == 0 ]]; then
    train_ckpt="latest" # Evaluate latest policy checkpoint.
else
    train_ckpt="0 1 2" # Evaluate multiple checkpoints for Demo-SCORE.
fi

# Dates.
train_filter=0  # Official: Used for demo filtering experiments.
train_select=1  # Official: Used for demo selection experiments.
if [[ $train_filter == 1 ]]; then
    date="102025"
    train_date="102025"
elif [[ $train_select == 1 ]]; then
    date="102025"
    train_date="102025"
else
    echo "Select a dataset setting."
    exit 1
fi

######################## Standard policy eval. ########################

# Lowdim-state-based experiments.
# state="low_dim"
# eval_save_episodes_over_seeds

# Image-state-based experiments.
state="image"
eval_save_episodes_over_seeds

# If an INT_ID was requested but there were fewer than N commands, return "NULL".
if [[ -n "${INT_ID}" && "${FOUND_NTH}" -eq 0 ]]; then
    echo "NULL"
fi