#!/bin/bash
#SBATCH -J CoRSAL_ASR  # Job name
#SBATCH -p general                    # Partition name (use "general" or appropriate partition)
#SBATCH -o baseline_initial_%j.txt    # Standard output file with job ID
#SBATCH -e baseline_initial_%j.err    # Standard error file with job ID
#SBATCH --mail-type=ALL               # Email notifications for all job events
#SBATCH --mail-user=mealieff@iu.edu   # Email address for notifications
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=1           # Number of tasks per node
#SBATCH --time=2-02:00:00             # Maximum run time (2 days, 2 hours)
#SBATCH --mem=16G                     # Memory allocation (16 GB)
#SBATCH -A r00018                     # SLURM account name

# Load the conda env
module load conda
conda activate ASRWORK

# Load the Python module
module load python/3.11.5

cd ~/tonal-transfer/models

python3

cd ~/N/project/CoRSAL 
