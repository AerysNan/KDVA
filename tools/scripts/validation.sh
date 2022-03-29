for stream in $(seq $2 1 $3)
do
  for retrain in $(seq 0 1 6)
  do
    mkdir -p snapshot/result/$1_${stream}_${retrain}_v$5
  done
  for epoch in $(seq 1 1 11)
  do
    CUDA_VISIBLE_DEVICES=$4 python3 tools/model_test.py configs/custom/ssd_base.py checkpoints/ssd.pth -d $1_${stream}_1_val_$(expr ${epoch} - 1) --out snapshot/result/$1_${stream}_0_v$5/$(printf %02d ${epoch}).pkl
  done
  for epoch in $(seq 1 1 11)
  do
    for retrain in $(seq 1 1 6)
    do
      CUDA_VISIBLE_DEVICES=$4 python3 tools/model_test.py configs/custom/ssd_base.py snapshot/models/$1_${stream}_${retrain}_$5/${epoch}.pth -d $1_${stream}_${retrain}_val_$(expr ${epoch} - 1) --out snapshot/result/$1_${stream}_${retrain}_v$5/$(printf %02d ${epoch}).pkl
    done
  done
done