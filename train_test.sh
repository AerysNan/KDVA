PREFIX=$1
SIZE=$2
DEVICE=$3

echo processing dataset ${PREFIX} with size ${SIZE}

rm -rf tmp_${PREFIX}
mkdir tmp_${PREFIX}
cp checkpoints/ssdlite_mobilenetv2_scratch_600e_coco_20210629_110627-974d9307.pth tmp_${PREFIX}/previous.pth
CUDA_VISIBLE_DEVICES=${DEVICE} python3 tools/test.py /home/ubuntu/urban/configs/custom/ssd_${PREFIX}_0.py /home/ubuntu/urban/tmp_${PREFIX}/previous.pth --eval bbox --out tmp_${PREFIX}/result_0.pkl
for i in $(seq 1 1 ${SIZE})
do
  CUDA_VISIBLE_DEVICES=${DEVICE} python3 tools/train.py /home/ubuntu/urban/configs/custom/ssd_${PREFIX}_${i}.py --work-dir tmp_${PREFIX}/
  mv tmp_${PREFIX}/latest.pth tmp_${PREFIX}/previous.pth
  CUDA_VISIBLE_DEVICES=${DEVICE} python3 tools/test.py  /home/ubuntu/urban/configs/custom/ssd_${PREFIX}_${i}.py /home/ubuntu/urban/tmp_${PREFIX}/previous.pth --eval bbox --out tmp_${PREFIX}/result_${i}.pkl
done
