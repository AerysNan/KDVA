for stream in $(seq $1 1 $2)
do
  for retrain in 020 040 060 080 100
  do
    mkdir -p snapshot/result/sub_${stream}_${retrain}_v$4
    for epoch in $(seq 1 1 11)
    do
      CUDA_VISIBLE_DEVICES=$3 python3 tools/test.py configs/custom/ssd_base.py snapshot/models/sub_${stream}_${retrain}_$4/${epoch}.pth -d sub_${stream}_${retrain}_val_$(expr ${epoch} - 1) --out snapshot/result/sub_${stream}_${retrain}_v$4/$(printf %02d ${epoch}).pkl
    done
  done
done