#!/bin/bash
#SBATCH --time=3:10:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=10
#SBATCH --mem=5000
#SBATCH --job-name=evaluate_translation
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --mail-user=p278658@rug.nl
#SBATCH --output=eva_015-silver-ckpt-4240-%j.log

#module load Python/2.7.11-foss-2016a
#module load tensorflow/0.10.0-foss-2016a-Python-2.7.11
#nohup python translate.py --evaluate --data_dir tmp_004 --train_dir chkpnt_004 > eva_nohup.out 2> eva_nohup.err < /dev/null &
#python translate.py --evaluate --data_dir tmp_004 --train_dir chkpnt_004
#nohup python translate.py --evaluate --data_dir ../models/tmp_009 --train_dir ../models/chkpnt_009 > eva_nohup_009.out 2> eva_nohup_009.err < /dev/null &
#nohup python translate.py --evaluate --data_dir ../models/tmp_010 --train_dir ../models/chkpnt_010 > eva_nohup_010.out 2> eva_nohup_010.err < /dev/null &
#python translate.py --evaluate --data_dir ../models/tmp_010 --train_dir ../models/chkpnt_010
#python translate.py --evaluate --data_dir ../models/tmp_013 --train_dir ../models/chkpnt_013
#python translate.py --evaluate --data_dir ../models/tmp_015 --train_dir ../models/chkpnt_015
#ython translate.py --evaluate --data_dir ../../../../models/tmp_017 --train_dir ../../../../models/chkpnt_017 > eva_017.log

#python translate.py --evaluate --data_dir ../../../../models/tmp_018 --train_dir ../../../../models/chkpnt_018 --test_files_prefix numtimegpoevetamdacdlcu-text_0.1_0.1_all-ws.test > eva_artificial_018.log 
python translate.py --evaluate --data_dir ../../../../models/tmp_018 --train_dir ../../../../models/chkpnt_018 --test_files_prefix pmb_2016-12-14_has_at_least_one_semorsymbow_filtered.test > eva_silver_018.log 
python translate.py --evaluate --data_dir ../../../../models/tmp_018 --train_dir ../../../../models/chkpnt_018 --test_files_prefix pmb_2016-12-14_sembows_only_filtered > eva_gold_018.log 
