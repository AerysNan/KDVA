DATAPATH=/home/ubuntu/data

PREFIX=$1
SIZE=$2
POSTFIX=$3
CONFIG=bh
DEVICE=$4

echo processing dataset ${PREFIX}_${POSTFIX}_${CONFIG} with size ${SIZE}

rm -rf ${DATAPATH}/tmp_${PREFIX}_${POSTFIX}_${CONFIG}
rm -rf ${DATAPATH}/tmp_${PREFIX}_${POSTFIX}_${CONFIG}_backbone
rm -rf ${DATAPATH}/snapshot/models/${PREFIX}_${POSTFIX}_${CONFIG}
rm -rf ${DATAPATH}/snapshot/result/${PREFIX}_${POSTFIX}_${CONFIG}

mkdir ${DATAPATH}/tmp_${PREFIX}_${POSTFIX}_${CONFIG}
mkdir ${DATAPATH}/tmp_${PREFIX}_${POSTFIX}_${CONFIG}_backbone
mkdir -p ${DATAPATH}/snapshot/models/${PREFIX}_${POSTFIX}_${CONFIG}
mkdir -p ${DATAPATH}/snapshot/result/${PREFIX}_${POSTFIX}_${CONFIG}

cp ${DATAPATH}/checkpoints/ssd.pth ${DATAPATH}/tmp_${PREFIX}_${POSTFIX}_${CONFIG}/latest.pth
cp ${DATAPATH}/checkpoints/ssd.pth ${DATAPATH}/tmp_${PREFIX}_${POSTFIX}_${CONFIG}_backbone/latest.pth
cp ${DATAPATH}/checkpoints/ssd.pth ${DATAPATH}/snapshot/models/${PREFIX}_${POSTFIX}_${CONFIG}/0.pth
CUDA_VISIBLE_DEVICES=${DEVICE} python3 tools/model_test.py configs/custom/ssd_na.py ${DATAPATH}/tmp_${PREFIX}_${POSTFIX}_${CONFIG}/latest.pth --out ${DATAPATH}/tmp_${PREFIX}_${POSTFIX}_${CONFIG}/$(printf %02d 0).pkl -d ${PREFIX}_test_0 -p ${DATAPATH}
for i in $(seq 1 1 $(expr ${SIZE} - 1))
do
  if [ $(expr ${i} % 3) -eq 0 ]; then
    CUDA_VISIBLE_DEVICES=${DEVICE} python3 tools/model_train.py configs/custom/ssd_fb.py --work-dir ${DATAPATH}/tmp_${PREFIX}_${POSTFIX}_${CONFIG}_backbone/ --train-dataset ${PREFIX}_2-1500_train_$(expr ${i} / 3 - 1) --no-validate --seed 0 --deterministic -p ${DATAPATH} --load-from ${DATAPATH}/checkpoints/ssd.pth
  fi
  CUDA_VISIBLE_DEVICES=${DEVICE} python3 tools/model_train.py configs/custom/ssd_fh.py --work-dir ${DATAPATH}/tmp_${PREFIX}_${POSTFIX}_${CONFIG}/ --train-dataset ${PREFIX}_${POSTFIX}_train_$(expr ${i} - 1) --no-validate --seed 0 --deterministic -p ${DATAPATH} --load-from ${DATAPATH}/tmp_${PREFIX}_${POSTFIX}_${CONFIG}_backbone/latest.pth
  cp ${DATAPATH}/tmp_${PREFIX}_${POSTFIX}_${CONFIG}/latest.pth ${DATAPATH}/snapshot/models/${PREFIX}_${POSTFIX}_${CONFIG}/${i}.pth
  CUDA_VISIBLE_DEVICES=${DEVICE} python3 tools/model_test.py configs/custom/ssd_na.py ${DATAPATH}/tmp_${PREFIX}_${POSTFIX}_${CONFIG}/latest.pth --out ${DATAPATH}/tmp_${PREFIX}_${POSTFIX}_${CONFIG}/$(printf %02d ${i}).pkl -d ${PREFIX}_test_${i} -p ${DATAPATH}
done

for i in $(seq 0 1 $(expr ${SIZE} - 1))
do
  cp ${DATAPATH}/tmp_${PREFIX}_${POSTFIX}_${CONFIG}/$(printf %02d ${i}).pkl ${DATAPATH}/snapshot/result/${PREFIX}_${POSTFIX}_${CONFIG}/$(printf %02d ${i}).pkl
done