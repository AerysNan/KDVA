PREFIX=$1
SIZE=$2
DEVICE=$3
POSTFIX=$4

echo processing dataset ${PREFIX}_${POSTFIX} with size ${SIZE}

rm -rf tmp_${PREFIX}_${POSTFIX}
rm -rf snapshot/models/${PREFIX}_${POSTFIX}
rm -rf snapshot/result/${PREFIX}_${POSTFIX}

mkdir tmp_${PREFIX}_${POSTFIX}
mkdir -p snapshot/models/${PREFIX}_${POSTFIX}
mkdir -p snapshot/result/${PREFIX}_${POSTFIX}

cp checkpoints/ssdlite_mobilenetv2_scratch_600e_coco_20210629_110627-974d9307.pth tmp_${PREFIX}_${POSTFIX}/latest.pth
cp checkpoints/ssdlite_mobilenetv2_scratch_600e_coco_20210629_110627-974d9307.pth snapshot/models/${PREFIX}_${POSTFIX}/0.pth
CUDA_VISIBLE_DEVICES=${DEVICE} python3 tools/test.py configs/custom/ssd.py tmp_${PREFIX}_${POSTFIX}/latest.pth --out tmp_${PREFIX}_${POSTFIX}/result_0.pkl -d ${PREFIX}_${POSTFIX}_test_0
for i in $(seq 1 1 $(expr ${SIZE} - 1))
do
  CUDA_VISIBLE_DEVICES=${DEVICE} python3 tools/train.py configs/custom/ssd.py --work-dir tmp_${PREFIX}_${POSTFIX}/ --train-dataset ${PREFIX}_${POSTFIX}_train_$(expr ${i} - 1) --val-dataset ${PREFIX}_${POSTFIX}_val_$(expr ${i} - 1)
  cp tmp_${PREFIX}_${POSTFIX}/latest.pth snapshot/models/${PREFIX}_${POSTFIX}/${i}.pth
  CUDA_VISIBLE_DEVICES=${DEVICE} python3 tools/test.py configs/custom/ssd.py tmp_${PREFIX}_${POSTFIX}/latest.pth --out tmp_${PREFIX}_${POSTFIX}/result_${i}.pkl -d ${PREFIX}_${POSTFIX}_test_${i}
done

CUDA_VISIBLE_DEVICES=${DEVICE} python3 tools/merge_result.py -d tmp_${PREFIX}_${POSTFIX} -c ${SIZE} -p result -f pkl
CUDA_VISIBLE_DEVICES=${DEVICE} python3 tools/expand_result.py -r tmp_${PREFIX}_${POSTFIX}/result.pkl -d snapshot/result/${PREFIX}_${POSTFIX}
CUDA_VISIBLE_DEVICES=${DEVICE} python3 tools/range_eval.py -p snapshot/result/${PREFIX}_${POSTFIX} -d ${PREFIX}_${POSTFIX} -s True -n ${SIZE}