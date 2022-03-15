PREFIX=$1
SIZE=$2
DEVICE=$3
POSTFIX=$4
CONFIG=$5

echo processing dataset ${PREFIX}_${POSTFIX}_${CONFIG} with size ${SIZE}

rm -rf tmp_${PREFIX}_${POSTFIX}_${CONFIG}
rm -rf snapshot/models/${PREFIX}_${POSTFIX}_${CONFIG}
rm -rf snapshot/result/${PREFIX}_${POSTFIX}_${CONFIG}

mkdir tmp_${PREFIX}_${POSTFIX}_${CONFIG}
mkdir -p snapshot/models/${PREFIX}_${POSTFIX}_${CONFIG}
mkdir -p snapshot/result/${PREFIX}_${POSTFIX}_${CONFIG}

cp checkpoints/ssd.pth tmp_${PREFIX}_${POSTFIX}_${CONFIG}/latest.pth
cp checkpoints/ssd.pth snapshot/models/${PREFIX}_${POSTFIX}_${CONFIG}/0.pth
CUDA_VISIBLE_DEVICES=${DEVICE} python3 tools/model_test.py configs/custom/ssd_${CONFIG}.py tmp_${PREFIX}_${POSTFIX}_${CONFIG}/latest.pth --out tmp_${PREFIX}_${POSTFIX}_${CONFIG}/$(printf %02d 0).pkl -d ${PREFIX}_test_0
for i in $(seq 1 1 $(expr ${SIZE} - 1))
do
  CUDA_VISIBLE_DEVICES=${DEVICE} python3 tools/model_train.py configs/custom/ssd_${CONFIG}.py --work-dir tmp_${PREFIX}_${POSTFIX}_${CONFIG}/ --train-dataset ${PREFIX}_${POSTFIX}_train_$(expr ${i} - 1) --val-dataset ${PREFIX}_${POSTFIX}_val_$(expr ${i} - 1) --no-test --load-from tmp_${PREFIX}_${POSTFIX}_${CONFIG}/latest.pth --seed 0 --deterministic
  cp tmp_${PREFIX}_${POSTFIX}_${CONFIG}/latest.pth snapshot/models/${PREFIX}_${POSTFIX}_${CONFIG}/${i}.pth
  CUDA_VISIBLE_DEVICES=${DEVICE} python3 tools/model_test.py configs/custom/ssd_${CONFIG}.py tmp_${PREFIX}_${POSTFIX}_${CONFIG}/latest.pth --out tmp_${PREFIX}_${POSTFIX}_${CONFIG}/$(printf %02d ${i}).pkl -d ${PREFIX}_test_${i}
done

CUDA_VISIBLE_DEVICES=${DEVICE} python3 tools/merge_result.py -d tmp_${PREFIX}_${POSTFIX}_${CONFIG} -o snapshot/merge/${PREFIX}_${POSTFIX}_${CONFIG}.pkl
for i in $(seq 0 1 $(expr ${SIZE} - 1))
do
  cp tmp_${PREFIX}_${POSTFIX}_${CONFIG}/$(printf %02d ${i}).pkl snapshot/result/${PREFIX}_${POSTFIX}_${CONFIG}/$(printf %02d ${i}).pkl
done
CUDA_VISIBLE_DEVICES=${DEVICE} python3 tools/range_eval.py -r ${PREFIX}_${POSTFIX}_${CONFIG} -d ${PREFIX} -s True -n ${SIZE} --gt False