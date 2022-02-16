PREFIX=$1
SIZE=$2
DEVICE=$3
POSTFIX=$4
CONFIG=$5

echo processing dataset ${PREFIX}_${POSTFIX}_${CONFIG} with size ${SIZE}

rm -rf tmp_acc_${PREFIX}_${POSTFIX}_${CONFIG}
rm -rf snapshot/models/${PREFIX}_${POSTFIX}_${CONFIG}_acc
rm -rf snapshot/result/${PREFIX}_${POSTFIX}_${CONFIG}_acc

mkdir tmp_acc_${PREFIX}_${POSTFIX}_${CONFIG}
mkdir -p snapshot/models/${PREFIX}_${POSTFIX}_${CONFIG}_acc
mkdir -p snapshot/result/${PREFIX}_${POSTFIX}_${CONFIG}_acc

cp checkpoints/ssd.pth tmp_acc_${PREFIX}_${POSTFIX}_${CONFIG}/latest.pth
cp checkpoints/ssd.pth snapshot/models/${PREFIX}_${POSTFIX}_${CONFIG}_acc/0.pth
CUDA_VISIBLE_DEVICES=${DEVICE} python3 tools/test.py configs/custom/ssd_${CONFIG}.py tmp_acc_${PREFIX}_${POSTFIX}_${CONFIG}/latest.pth --out tmp_acc_${PREFIX}_${POSTFIX}_${CONFIG}/$(printf %02d 0).pkl -d ${PREFIX}_${POSTFIX}_test_0
for i in $(seq 1 1 $(expr ${SIZE} - 1))
do
  CUDA_VISIBLE_DEVICES=${DEVICE} python3 tools/train.py configs/custom/ssd_${CONFIG}.py --work-dir tmp_acc_${PREFIX}_${POSTFIX}_${CONFIG}/ --train-dataset ${PREFIX}_${POSTFIX}_train_$(expr ${i} - 1) --val-dataset ${PREFIX}_${POSTFIX}_test_${i} --load-from tmp_acc_${PREFIX}_${POSTFIX}_${CONFIG}/latest.pth --no-validate
  cp tmp_acc_${PREFIX}_${POSTFIX}_${CONFIG}/latest.pth snapshot/models/${PREFIX}_${POSTFIX}_${CONFIG}_acc/${i}.pth
  CUDA_VISIBLE_DEVICES=${DEVICE} python3 tools/test.py configs/custom/ssd_${CONFIG}.py tmp_acc_${PREFIX}_${POSTFIX}_${CONFIG}/latest.pth --out tmp_acc_${PREFIX}_${POSTFIX}_${CONFIG}/$(printf %02d ${i}).pkl -d ${PREFIX}_${POSTFIX}_test_${i}
done

CUDA_VISIBLE_DEVICES=${DEVICE} python3 tools/merge_result.py -d tmp_acc_${PREFIX}_${POSTFIX}_${CONFIG} -o snapshot/merge/${PREFIX}_${POSTFIX}_${CONFIG}_acc.pkl
for i in $(seq 0 1 $(expr ${SIZE} - 1))
do
  mv tmp_acc_${PREFIX}_${POSTFIX}_${CONFIG}/$(printf %02d ${i}).pkl snapshot/result/${PREFIX}_${POSTFIX}_${CONFIG}_acc/$(printf %02d ${i}).pkl
done
CUDA_VISIBLE_DEVICES=${DEVICE} python3 tools/range_eval.py -r ${PREFIX}_${POSTFIX}_${CONFIG}_acc -d ${PREFIX}_${POSTFIX} -s True -n ${SIZE}