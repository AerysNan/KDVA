PREFIX=$1
SIZE=$2
POSTFIX=$3
CONFIG=$6
DEVICE=$7

for stream in $4 $5
do
  rm -rf tmp_merge_${PREFIX}_${stream}m$4$5_${POSTFIX}_${CONFIG}
  rm -rf tmp_merge_${PREFIX}_merge$4$5_${POSTFIX}_${CONFIG}
  rm -rf snapshot/models/${PREFIX}_${stream}m$4$5_${POSTFIX}_${CONFIG}_merge
  rm -rf snapshot/result/${PREFIX}_${stream}m$4$5_${POSTFIX}_${CONFIG}_merge

  mkdir tmp_merge_${PREFIX}_${stream}m$4$5_${POSTFIX}_${CONFIG}
  mkdir tmp_merge_${PREFIX}_merge$4$5_${POSTFIX}_${CONFIG}
  mkdir -p snapshot/models/${PREFIX}_${stream}m$4$5_${POSTFIX}_${CONFIG}_merge
  mkdir -p snapshot/result/${PREFIX}_${stream}m$4$5_${POSTFIX}_${CONFIG}_merge

  cp checkpoints/ssd.pth tmp_merge_${PREFIX}_merge$4$5_${POSTFIX}_${CONFIG}/latest.pth
  cp checkpoints/ssd.pth snapshot/models/${PREFIX}_${stream}m$4$5_${POSTFIX}_${CONFIG}_merge/0.pth
done

for stream in $4 $5
do
  cp tmp_merge_${PREFIX}_merge$4$5_${POSTFIX}_${CONFIG}/latest.pth tmp_merge_${PREFIX}_${stream}m$4$5_${POSTFIX}_${CONFIG}/latest.pth
  CUDA_VISIBLE_DEVICES=${DEVICE} python3 tools/test.py configs/custom/ssd_base.py tmp_merge_${PREFIX}_${stream}m$4$5_${POSTFIX}_${CONFIG}/latest.pth --out tmp_merge_${PREFIX}_${stream}m$4$5_${POSTFIX}_${CONFIG}/$(printf %02d 0).pkl -d ${PREFIX}_${stream}_test_0
done

for i in $(seq 1 1 $(expr ${SIZE} - 1))
do
  CUDA_VISIBLE_DEVICES=${DEVICE} python3 tools/train.py configs/custom/ssd_${CONFIG}.py --work-dir tmp_merge_${PREFIX}_merge$4$5_${POSTFIX}_${CONFIG}/ --train-dataset merge$4$5_${POSTFIX}_train_$(expr ${i} - 1) --no-validate --load-from tmp_merge_${PREFIX}_merge$4$5_${POSTFIX}_${CONFIG}/latest.pth --seed 0 --deterministic
  for stream in $4 $5
  do
    cp tmp_merge_${PREFIX}_merge$4$5_${POSTFIX}_${CONFIG}/latest.pth tmp_merge_${PREFIX}_${stream}m$4$5_${POSTFIX}_${CONFIG}/latest.pth
    cp tmp_merge_${PREFIX}_merge$4$5_${POSTFIX}_${CONFIG}/latest.pth snapshot/models/${PREFIX}_${stream}m$4$5_${POSTFIX}_${CONFIG}_merge/${i}.pth
  done
  
  for stream in $4 $5
  do
    CUDA_VISIBLE_DEVICES=${DEVICE} python3 tools/test.py configs/custom/ssd_base.py tmp_merge_${PREFIX}_${stream}m$4$5_${POSTFIX}_${CONFIG}/latest.pth --out tmp_merge_${PREFIX}_${stream}m$4$5_${POSTFIX}_${CONFIG}/$(printf %02d ${i}).pkl -d ${PREFIX}_${stream}_test_${i}
  done
done

for stream in $4 $5
do
  python3 tools/merge_result.py -d tmp_merge_${PREFIX}_${stream}m$4$5_${POSTFIX}_${CONFIG} -o snapshot/merge/${PREFIX}_${stream}m$4$5_${POSTFIX}_${CONFIG}_merge.pkl
done

for stream in $4 $5
do
  for i in $(seq 0 1 $(expr ${SIZE} - 1))
  do
    cp tmp_merge_${PREFIX}_${stream}m$4$5_${POSTFIX}_${CONFIG}/$(printf %02d ${i}).pkl snapshot/result/${PREFIX}_${stream}m$4$5_${POSTFIX}_${CONFIG}_merge/$(printf %02d ${i}).pkl
  done
done

for stream in $4 $5
do
  echo range eval ${stream}
  python3 tools/range_eval.py -r ${PREFIX}_${stream}m$4$5_${POSTFIX}_${CONFIG}_merge -d ${PREFIX}_${stream} -s True -n ${SIZE} --gt False > ${stream}m$4$5_${POSTFIX}_${CONFIG}_merge.txt
done