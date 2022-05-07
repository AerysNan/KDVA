for retrain in $(seq 1 1 6)
do
  for epoch in $(seq 0 1 11)
  do
    rm -rf tmp.txt
    echo detrac_$1_${retrain}_train_${epoch} >> tmp.txt
    echo detrac_$2_${retrain}_train_${epoch} >> tmp.txt
    python3 tools/merge_traces.py -r /home/ubuntu/data -o merge$1$2_${retrain}_train_${epoch} -i tmp.txt -g False -a True
    rm -rf tmp.txt
  done
done