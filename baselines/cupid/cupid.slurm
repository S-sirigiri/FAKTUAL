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
## sbatch cupid.slurm 

## Slurm parameters
##SBATCH --ntasks-per-node=248
##SBATCH --ntasks=1  # this should be equal to 1 as we are using array
#SBATCH --requeue
#SBATCH --array=1-20%10  # number tasks to execute. The total number of trajectories generated is this number multiplied by BATCH_TRAJECTORIES. %20 means the number of jobs running simultaneously.
#SBATCH --cpus-per-task=30
#SBATCH --account=nvr_srl_simpler
#SBATCH --gpus=1
#SBATCH --job-name="cupid"
##SBATCH --partition=interactive,backfill_block1,backfill_singlenode  # interactive, grizzly for nvr resources, polar for general resources, longrun_grizzly for longer runs
#SBATCH --partition=grizzly,polar,polar3,polar4,batch_singlenode
#SBATCH --time=0-04:00:00
#SBATCH --open-mode=append
#SBATCH --signal=B:TERM@15  # tells the controller
                            # to send SIGTERM to the job 15 secs
                            # before its time ends to give it a
                            # chance for better cleanup.
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/ftozetoramos/slurm_outputs/slurm_%A_%a.out

export NUM_TASKS=$SLURM_ARRAY_TASK_COUNT
export CODE_DIR=""
export SEED=$RANDOM
export CONFIG_NAME="procedural_kitchen_open_dishwasher"

# Trap SIGTERM signals and resubmit the job if it is killed by reaching max time limit
trap 'scontrol requeue ${SLURM_JOB_ID}; exit 15' SIGTERM

## Run in parallel using array 
echo "==================== generating data  ======================="

  srun \
    --container-image=/lustre/fsw/portfolios/nvr/projects/nvr_srl_simpler/images/cupid.sqsh \
    --container-name=cupid_updated_${SLURM_JOB_ID} \
    --container-writable \
    --ntasks=1 \
    --gpus=1 \
    --cpus-per-task=5 \
    --container-mounts=$HOME:/home,/lustre:/lustre \
    /bin/bash -c  "pip3 install --no-build-isolation -e TODO"

  srun \
    --container-name=cupid_updated_${SLURM_JOB_ID} \
    --container-writable \
    --container-entrypoint \
    --container-mounts=$HOME:/home,/lustre:/lustre \
    --ntasks=1 \
    --gpus=1 \
    --time=04:00:00 \
    --cpus-per-task=$SLURM_CPUS_PER_TASK \
    /bin/bash -c "\
          echo -n 'Output directory: ' && echo $OUTPUT_DIR && \
          /lustre/fsw/portfolios/nvr/users/ftozetoramos/code/robo_dataset_diversity/baselines/cupid/scripts/train/train_policies.sh $SEED ${OUTPUT_DIR} ${PARAMETERS}" &
wait

## Copy to Swiftstack
#srun \
#  /bin/bash -c "~/bin/aws/dist/aws s3 --endpoint-url https://pbss.s8k.io --region us-east-1 sync $OUTPUT_DIR $SWIFTSTACK_DIR"



