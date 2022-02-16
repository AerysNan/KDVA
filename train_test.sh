# PREFIX=$1
# SIZE=$2
# DEVICE=$3
# POSTFIX=$4
# CONFIG=$5
# EPOCH=200

# echo processing dataset ${PREFIX}_${POSTFIX}_${CONFIG} with size ${SIZE}

# rm -rf tmp_${PREFIX}_${POSTFIX}_${CONFIG}
# rm -rf snapshot/models/${PREFIX}_${POSTFIX}_${CONFIG}
# rm -rf snapshot/result/${PREFIX}_${POSTFIX}_${CONFIG}

# mkdir tmp_${PREFIX}_${POSTFIX}_${CONFIG}
# CUDA_VISIBLE_DEVICES=${DEVICE} python3 tools/test.py configs/custom/ssd_${CONFIG}.py checkpoints/ssd.pth --out tmp_${PREFIX}_${POSTFIX}_${CONFIG}/00.pkl -d ${PREFIX}_${POSTFIX}_test_0

# for e in $(seq 10 10 ${EPOCH})
# do
#   mkdir -p snapshot/models/${PREFIX}_${POSTFIX}_${CONFIG}/${e}
#   mkdir -p snapshot/result/${PREFIX}_${POSTFIX}_${CONFIG}/${e}
#   cp tmp_${PREFIX}_${POSTFIX}_${CONFIG}/$(printf %02d 0).pkl snapshot/result/${PREFIX}_${POSTFIX}_${CONFIG}/${e}/00.pkl
#   cp checkpoints/ssd.pth snapshot/models/${PREFIX}_${POSTFIX}_${CONFIG}/${e}/0.pkl
# done

# for i in $(seq 1 1 $(expr ${SIZE} - 1))
# do
#   CUDA_VISIBLE_DEVICES=${DEVICE} python3 tools/train.py configs/custom/ssd_${CONFIG}.py --work-dir tmp_${PREFIX}_${POSTFIX}_${CONFIG}/ --train-dataset ${PREFIX}_${POSTFIX}_train_$(expr ${i} - 1) --val-dataset ${PREFIX}_${POSTFIX}_test_${i} --no-validate
#   for e in $(seq 10 10 ${EPOCH})
#   do
#     cp tmp_${PREFIX}_${POSTFIX}_${CONFIG}/epoch_${e}.pth snapshot/models/${PREFIX}_${POSTFIX}_${CONFIG}/${e}/${i}.pth
#     CUDA_VISIBLE_DEVICES=${DEVICE} python3 tools/test.py configs/custom/ssd_${CONFIG}.py snapshot/models/${PREFIX}_${POSTFIX}_${CONFIG}/${e}/${i}.pth --out snapshot/result/${PREFIX}_${POSTFIX}_${CONFIG}/${e}/$(printf %02d ${i}).pkl -d ${PREFIX}_${POSTFIX}_test_${i}
#   done
# done

# for e in $(seq 10 10 ${EPOCH})
# do
#   CUDA_VISIBLE_DEVICES=${DEVICE} python3 tools/merge_result.py -d snapshot/result/${PREFIX}_${POSTFIX}_${CONFIG}/${e} -o snapshot/merge/${PREFIX}_${POSTFIX}_${CONFIG}_${e}.pkl
#   echo epoch: ${e}
#   CUDA_VISIBLE_DEVICES=${DEVICE} python3 tools/range_eval.py -r ${PREFIX}_${POSTFIX}_${CONFIG} -d ${PREFIX}_${POSTFIX} -s True -n ${SIZE} -p ${e}
# done


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
CUDA_VISIBLE_DEVICES=${DEVICE} python3 tools/test.py configs/custom/ssd_${CONFIG}.py tmp_${PREFIX}_${POSTFIX}_${CONFIG}/latest.pth --out tmp_${PREFIX}_${POSTFIX}_${CONFIG}/$(printf %02d 0).pkl -d ${PREFIX}_${POSTFIX}_test_0
for i in $(seq 1 1 $(expr ${SIZE} - 1))
do
  CUDA_VISIBLE_DEVICES=${DEVICE} python3 tools/train.py configs/custom/ssd_${CONFIG}.py --work-dir tmp_${PREFIX}_${POSTFIX}_${CONFIG}/ --train-dataset ${PREFIX}_${POSTFIX}_train_$(expr ${i} - 1) --val-dataset ${PREFIX}_${POSTFIX}_test_${i}
  cp tmp_${PREFIX}_${POSTFIX}_${CONFIG}/latest.pth snapshot/models/${PREFIX}_${POSTFIX}_${CONFIG}/${i}.pth
  CUDA_VISIBLE_DEVICES=${DEVICE} python3 tools/test.py configs/custom/ssd_${CONFIG}.py tmp_${PREFIX}_${POSTFIX}_${CONFIG}/latest.pth --out tmp_${PREFIX}_${POSTFIX}_${CONFIG}/$(printf %02d ${i}).pkl -d ${PREFIX}_${POSTFIX}_test_${i}
done

CUDA_VISIBLE_DEVICES=${DEVICE} python3 tools/merge_result.py -d tmp_${PREFIX}_${POSTFIX}_${CONFIG} -o snapshot/merge/${PREFIX}_${POSTFIX}_${CONFIG}.pkl
for i in $(seq 0 1 $(expr ${SIZE} - 1))
do
  mv tmp_${PREFIX}_${POSTFIX}_${CONFIG}/$(printf %02d ${i}).pkl snapshot/result/${PREFIX}_${POSTFIX}_${CONFIG}/$(printf %02d ${i}).pkl
done
CUDA_VISIBLE_DEVICES=${DEVICE} python3 tools/range_eval.py -r ${PREFIX}_${POSTFIX}_${CONFIG} -d ${PREFIX}_${POSTFIX} -s True -n ${SIZE}