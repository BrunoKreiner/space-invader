#!/bin/sh
#SBATCH --time=6:00:00 # never more than 6 hours
#SBATCH --cpus-per-task=16  # must match num_env, never more than 16
#SBATCH --gres=gpu:0  # never more than 0
#SBATCH --partition=performance # do not change
#SBATCH --output=out/rle-mini-challenge-%A_%a.out

singularity pull docker://yanickschraner/rle-mini-challenge
singularity exec -B ${HOME}/rle-assginment:${HOME}/rle-assginment rle-mini-challenge_latest.sif ${HOME}/rle-assginment/dqn_example.py --mode train --nocuda --num_envs 16 --total_steps 10000000