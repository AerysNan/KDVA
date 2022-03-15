for stream in $(seq $1 1 $2)
do
  for retrain in 020 040 060 080 100
  do
    mkdir -p snapshot/result/sub_${stream}_${retrain}_v$4
  done


  for epoch in $(seq 12 1 14)
  do
    # CUDA_VISIBLE_DEVICES=$3 python3 tools/model_test.py configs/custom/ssd_base.py checkpoints/ssd.pth -d virat_${stream}_020_val_$(expr ${epoch} - 1) --out snapshot/result/virat_${stream}_000_v$4/$(printf %02d ${epoch}).pkl

    for retrain in 020 040 060 080 100
    do
    
      CUDA_VISIBLE_DEVICES=$3 python3 tools/model_test.py configs/custom/ssd_base.py snapshot/models/virat_${stream}_${retrain}_$4/${epoch}.pth -d virat_${stream}_${retrain}_val_$(expr ${epoch} - 1) --out snapshot/result/virat_${stream}_${retrain}_v$4/$(printf %02d ${epoch}).pkl
    done
  done
done