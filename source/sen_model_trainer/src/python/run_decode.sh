#!/bin/bash
#SBATCH --time=20:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --job-name=python_cpu
#SBATCH --mem=1g
#module load Python/2.7.11-foss-2016a
#module load tensorflow/0.10.0-foss-2016a-Python-2.7.11
#nohup python translate.py --train --evaluate --data_dir tmp_004 --train_dir chkpnt_004 > nohup.out 2> nohup.err < /dev/null &
python translate.py --decode --data_dir ../../../../models/tmp_018 --train_dir ../../../../models/chkpnt_018
