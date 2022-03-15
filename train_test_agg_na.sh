PREFIX=$1
SIZE=$2
POSTFIX=$3
DEVICE=$6

for stream in $4 $5
do
  rm -rf tmp_agg_${PREFIX}_${stream}m$4$5_${POSTFIX}
  rm -rf tmp_agg_${PREFIX}_merge$4$5_${POSTFIX}
  rm -rf snapshot/models/${PREFIX}_${stream}m$4$5_${POSTFIX}_agg
  rm -rf snapshot/result/${PREFIX}_${stream}m$4$5_${POSTFIX}_agg

  mkdir tmp_agg_${PREFIX}_${stream}m$4$5_${POSTFIX}
  mkdir tmp_agg_${PREFIX}_merge$4$5_${POSTFIX}
  mkdir -p snapshot/models/${PREFIX}_${stream}m$4$5_${POSTFIX}_agg
  mkdir -p snapshot/result/${PREFIX}_${stream}m$4$5_${POSTFIX}_agg

  cp checkpoints/ssd.pth tmp_agg_${PREFIX}_merge$4$5_${POSTFIX}/latest.pth
  cp checkpoints/ssd.pth snapshot/models/${PREFIX}_${stream}m$4$5_${POSTFIX}_agg/0.pth
done

for stream in $4 $5
do
  cp tmp_agg_${PREFIX}_merge$4$5_${POSTFIX}/latest.pth tmp_agg_${PREFIX}_${stream}m$4$5_${POSTFIX}/latest.pth
  CUDA_VISIBLE_DEVICES=${DEVICE} python3 tools/model_test.py configs/custom/ssd_base.py tmp_agg_${PREFIX}_${stream}m$4$5_${POSTFIX}/latest.pth --out tmp_agg_${PREFIX}_${stream}m$4$5_${POSTFIX}/$(printf %02d 0).pkl -d ${PREFIX}_${stream}_test_0
done

for i in $(seq 1 1 $(expr ${SIZE} - 1))
do

  CUDA_VISIBLE_DEVICES=${DEVICE} python3 tools/model_train.py configs/custom/ssd_fh.py --work-dir tmp_agg_${PREFIX}_merge$4$5_${POSTFIX}/ --train-dataset merge$4$5_${POSTFIX}_train_$(expr ${i} - 1) --no-validate --seed 0 --deterministic
  for stream in $4 $5
  do
    cp tmp_agg_${PREFIX}_merge$4$5_${POSTFIX}/latest.pth tmp_agg_${PREFIX}_${stream}m$4$5_${POSTFIX}/latest.pth
  done
  
  for stream in $4 $5
  do
    CUDA_VISIBLE_DEVICES=${DEVICE} python3 tools/model_train.py configs/custom/ssd_fb.py --work-dir tmp_agg_${PREFIX}_${stream}m$4$5_${POSTFIX}/ --train-dataset ${PREFIX}_${stream}_${POSTFIX}_train_$(expr ${i} - 1) --no-validate --load-from tmp_agg_${PREFIX}_${stream}m$4$5_${POSTFIX}/latest.pth --seed 0 --deterministic
    cp tmp_agg_${PREFIX}_${stream}m$4$5_${POSTFIX}/latest.pth snapshot/models/${PREFIX}_${stream}m$4$5_${POSTFIX}_agg/${i}.pth
  done

  for stream in $4 $5
  do
    CUDA_VISIBLE_DEVICES=${DEVICE} python3 tools/model_test.py configs/custom/ssd_base.py tmp_agg_${PREFIX}_${stream}m$4$5_${POSTFIX}/latest.pth --out tmp_agg_${PREFIX}_${stream}m$4$5_${POSTFIX}/$(printf %02d ${i}).pkl -d ${PREFIX}_${stream}_test_${i}
  done
done

for stream in $4 $5
do
  python3 tools/merge_result.py -d tmp_agg_${PREFIX}_${stream}m$4$5_${POSTFIX} -o snapshot/merge/${PREFIX}_${stream}m$4$5_${POSTFIX}_agg.pkl
done

for stream in $4 $5
do
  for i in $(seq 0 1 $(expr ${SIZE} - 1))
  do
    cp tmp_agg_${PREFIX}_${stream}m$4$5_${POSTFIX}/$(printf %02d ${i}).pkl snapshot/result/${PREFIX}_${stream}m$4$5_${POSTFIX}_agg/$(printf %02d ${i}).pkl
  done
done

for stream in $4 $5
do
  echo range eval ${stream}
  python3 tools/range_eval.py -r ${PREFIX}_${stream}m$4$5_${POSTFIX}_agg -d ${PREFIX}_${stream} -s True -n ${SIZE} --gt False > ${stream}m$4$5_${POSTFIX}_agg.txt
done