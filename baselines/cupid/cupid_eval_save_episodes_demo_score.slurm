#!/bin/bash

#####################################################################################
## Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto. Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
#####################################################################################

## Usage:
## sbatch cupid_eval.slurm 

## Slurm parameters
##SBATCH --ntasks-per-node=248
##SBATCH --ntasks=1  # this should be equal to 1 as we are using array
#SBATCH --requeue
#SBATCH --array=1-36%36  # number tasks to execute. %10 is the number of tasks to execute simultaneously
#SBATCH --cpus-per-task=30
#SBATCH --account=nvr_srl_simpler
##SBATCH --gpus=1
#SBATCH --gres=gpu:1
##SBATCH --gpus-per-task=1
#SBATCH --job-name="cupid_eval"
##SBATCH --partition=interactive,backfill_block1,backfill_singlenode  # interactive, grizzly for nvr resources, polar for general resources, longrun_grizzly for longer runs
#SBATCH --partition=batch_singlenode,backfill_singlenode
#SBATCH --time=0-04:00:00
#SBATCH --open-mode=append
#SBATCH --signal=B:TERM@120  # tells the controller
                             # to send SIGTERM to the job 120 secs
                             # before its time ends to give it a
                             # chance for better cleanup.
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/ftozetoramos/slurm_outputs/slurm_%A_%a.out

export NUM_TASKS=$SLURM_ARRAY_TASK_COUNT
export CODE_DIR=""
export SCRIPT="/lustre/fsw/portfolios/nvr/users/ftozetoramos/code/robo_dataset_diversity/baselines/cupid/scripts/eval/eval_save_episodes_demo_score.sh"
#export TASK_IDS=(7 11 13 15 17 23 25 33)
export TASK_IDS=({1..36})

# Trap SIGTERM signals and resubmit the job if it is killed by reaching max time limit
trap 'scontrol requeue ${SLURM_JOB_ID}; exit 15' SIGTERM

## Run in parallel using array 
echo "==================== preparing the node  ======================="

  srun \
    --container-image=/lustre/fsw/portfolios/nvr/projects/nvr_srl_simpler/images/cupid.sqsh \
    --container-name=cupid_updated_${SLURM_JOB_ID} \
    --container-writable \
    --ntasks=1 \
    --gpus-per-task=1 \
    --cpus-per-task=5 \
    --container-mounts=$HOME:/home,/lustre:/lustre \
    /bin/bash -c  "ls"

  srun \
    --container-name=cupid_updated_${SLURM_JOB_ID} \
    --container-writable \
    --container-entrypoint \
    --container-mounts=$HOME:/home,/lustre:/lustre \
    --ntasks=1 \
    --gpus-per-task=1 \
    --time=04:00:00 \
    --cpus-per-task=$SLURM_CPUS_PER_TASK \
    /bin/bash -c "\
          INDEX=$((SLURM_ARRAY_TASK_ID - 1)) && \
          TASK=${TASK_IDS[$INDEX]} && \
          echo -n 'Running script: ' && echo $SCRIPT && echo ${TASK} && \
	  $($SCRIPT ${TASK})" &
wait

## Copy to Swiftstack
#srun \
#  /bin/bash -c "~/bin/aws/dist/aws s3 --endpoint-url https://pbss.s8k.io --region us-east-1 sync $OUTPUT_DIR $SWIFTSTACK_DIR"



