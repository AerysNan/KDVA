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
CUDA_VISIBLE_DEVICES=${DEVICE} python3 tools/test.py ${HOME}/urban/configs/custom/ssd_${PREFIX}_${POSTFIX}_0.py ${HOME}/urban/tmp_${PREFIX}_${POSTFIX}/latest.pth --out tmp_${PREFIX}_${POSTFIX}/result_0.pkl
for i in $(seq 1 1 $(expr ${SIZE} - 1))
do
  CUDA_VISIBLE_DEVICES=${DEVICE} python3 tools/train.py ${HOME}/urban/configs/custom/ssd_${PREFIX}_${POSTFIX}_${i}.py --work-dir tmp_${PREFIX}_${POSTFIX}/
  cp tmp_${PREFIX}_${POSTFIX}/latest.pth snapshot/models/${PREFIX}_${POSTFIX}/${i}.pth
  CUDA_VISIBLE_DEVICES=${DEVICE} python3 tools/test.py ${HOME}/urban/configs/custom/ssd_${PREFIX}_${POSTFIX}_${i}.py ${HOME}/urban/tmp_${PREFIX}_${POSTFIX}/latest.pth --out tmp_${PREFIX}_${POSTFIX}/result_${i}.pkl
done

CUDA_VISIBLE_DEVICES=${DEVICE} python3 tools/merge_result.py -d tmp_${PREFIX}_${POSTFIX} -c ${SIZE} -p result -f pkl
CUDA_VISIBLE_DEVICES=${DEVICE} python3 tools/expand_result.py -r tmp_${PREFIX}_${POSTFIX}/result.pkl -d snapshot/result/${PREFIX}_${POSTFIX}
CUDA_VISIBLE_DEVICES=${DEVICE} python3 tools/range_eval.py -p snapshot/result/${PREFIX}_${POSTFIX} -d ${PREFIX}_${POSTFIX} -s True -n ${SIZE}