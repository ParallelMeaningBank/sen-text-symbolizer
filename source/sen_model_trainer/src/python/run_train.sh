# Copy file from dataset to data folder
echo "Copy data from dataset to model folder."
mkdir ../../../../models/tmp_019
mkdir ../../../../models/chkpnt_019
cp -rp ../../../../dataset/019/* ../../../../models/tmp_019

# Now run the program
echo "Started training. See log file at train_019.log."
python translate.py --train --data_dir ../../../../models/tmp_019 --train_dir ../../../../models/chkpnt_019 --train_files_prefix combined_dataset_pmb_2016-12-14.train --dev_files_prefix combined_dataset_pmb_2016-12-14.dev > train_019.log 

