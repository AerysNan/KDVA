for retrain in 020 040 060 080 100
do
  for epoch in $(seq 0 1 11)
  do
    echo detrac_$1_${retrain}_train_${epoch} >> tmp.txt
    echo detrac_$2_${retrain}_train_${epoch} >> tmp.txt
    python3 tools/merge_traces.py -o merge$1$2_${retrain}_train_${epoch} -i tmp.txt -g False -a True
    rm -rf tmp.txt
  done
done