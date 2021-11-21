PREFIX=$1
SIZE=$2
CONFIG=$3
DEVICE=$4

echo processing dataset ${PREFIX} with size ${SIZE}

rm -rf tmp_${PREFIX}_${CONFIG}
mkdir tmp_${PREFIX}_${CONFIG}
mkdir -p models/${PREFIX}_${CONFIG}
cp checkpoints/ssdlite_mobilenetv2_scratch_600e_coco_20210629_110627-974d9307.pth tmp_${PREFIX}_${CONFIG}/latest.pth
CUDA_VISIBLE_DEVICES=${DEVICE} python3 tools/test.py ${HOME}/urban/configs/custom/ssd_${PREFIX}_0_${CONFIG}.py ${HOME}/urban/tmp_${PREFIX}_${CONFIG}/latest.pth --eval bbox --out tmp_${PREFIX}_${CONFIG}/result_0.pkl
for i in $(seq 1 1 $(expr ${SIZE} - 1))
do
  CUDA_VISIBLE_DEVICES=${DEVICE} python3 tools/train.py ${HOME}/urban/configs/custom/ssd_${PREFIX}_${i}_${CONFIG}.py --work-dir tmp_${PREFIX}_${CONFIG}/
  cp tmp_${PREFIX}_${CONFIG}/latest.pth models/${PREFIX}_${CONFIG}/${i}.pth
  CUDA_VISIBLE_DEVICES=${DEVICE} python3 tools/test.py ${HOME}/urban/configs/custom/ssd_${PREFIX}_${i}_${CONFIG}.py ${HOME}/urban/tmp_${PREFIX}_${CONFIG}/latest.pth --eval bbox --out tmp_${PREFIX}_${CONFIG}/result_${i}.pkl
done

CUDA_VISIBLE_DEVICES=${DEVICE} python3 tools/merge_result.py -d tmp_${PREFIX}_${CONFIG} -c ${SIZE} -p result -f pkl
CUDA_VISIBLE_DEVICES=${DEVICE} python3 tools/evaluate_from_file.py ${HOME}/urban/configs/ssd/ssdlite_mobilenetv2_scratch_600e_coco.py tmp_${PREFIX}_${CONFIG}/result.pkl -d ${PREFIX}