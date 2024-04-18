#!/bin/bash
#SBATCH --job-name=pyg-multinode-tutorial # identifier for the job listings
#SBATCH --output=pyg-multinode.log        # outputfile
#SBATCH --job-name=pyg-multinode-tutorial # identifier for the job listings
#SBATCH --output=pyg-multinode.log        # outputfile
#SBATCH --partition=debug             # ADJUST this to your system
#SBATCH -N 2                              # number of nodes you want to use
#SBATCH --ntasks=2                        # number of processes to be run
#SBATCH --gpus-per-task=1                 # every process wants one GPU!
                 # NCCL can't deal with task-binding...
## Now you can add more stuff for your convenience
            # total number of memory available per node (tensorflow need(ed) at least <GPU-memory> per GPU)
#SBATCH --export=ALL                      # use your shell environment (PATHs, ...)
#SBATCH --gres=gpu:1


# Thanks for shell-ideas to https://github.com/PrincetonUniversity/multi_gpu_training
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export OMPI_MCA_pml=ucx
echo "MASTER_ADDR:MASTER_PORT="${MASTER_ADDR}:${MASTER_PORT}

echo "###########################################################################"
echo "We recommend you set up your environment here (conda/spack/pip/modulefiles)"
echo "then remove --export=ALL (allows running the sbatch from any shell"
echo "###########################################################################"

# use --output=0 so that only the first task logs to the file!
srun python -m torch.distributed.launch start.py