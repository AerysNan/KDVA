PREFIX=$1
SIZE=$2
DEVICE=$3
CONFIG=$4

echo processing dataset ${PREFIX}_0 with size ${SIZE}

rm -rf snapshot/result/${PREFIX}_0
mkdir -p snapshot/result/${PREFIX}_0

for i in $(seq 0 1 $(expr ${SIZE} - 1))
do
  CUDA_VISIBLE_DEVICES=${DEVICE} python3 tools/model_test.py configs/custom/ssd_${CONFIG}.py checkpoints/ssd.pth --out snapshot/result/${PREFIX}_0/$(printf %02d ${i}).pkl -d ${PREFIX}_test_${i}
done