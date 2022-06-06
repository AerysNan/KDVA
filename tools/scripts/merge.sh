for retrain in $(seq 1 1 6)
do
  for epoch in $(seq 0 1 9)
  do
    rm -rf tmp.txt
    echo ut_$1_${retrain}-10800_train_${epoch} >> tmp.txt
    echo ut_$2_${retrain}-10800_train_${epoch} >> tmp.txt
    echo ut_$3_${retrain}-10800_train_${epoch} >> tmp.txt
    echo ut_$4_${retrain}-10800_train_${epoch} >> tmp.txt
    python3 tools/merge_traces.py -r /home/ubuntu/data -o ut$1$2$3$4_${retrain}-10800_train_${epoch} -i tmp.txt -g False -a True
    rm -rf tmp.txt
  done
done