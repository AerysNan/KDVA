DATAPATH=/home/ubuntu/data

PREFIX=$1
SIZE=$2
POSTFIX=$3
CONFIG=ms
DEVICE=$6

for stream in $4 $5
do
  rm -rf ${DATAPATH}/tmp_agg_${PREFIX}_${stream}_${POSTFIX}_${CONFIG}
  rm -rf ${DATAPATH}/tmp_agg_${PREFIX}_agg$4$5_${POSTFIX}_${CONFIG}
  rm -rf ${DATAPATH}/snapshot/models/${PREFIX}_${stream}_${POSTFIX}_${CONFIG}
  rm -rf ${DATAPATH}/snapshot/result/${PREFIX}_${stream}_${POSTFIX}_${CONFIG}

  mkdir ${DATAPATH}/tmp_agg_${PREFIX}_${stream}_${POSTFIX}_${CONFIG}
  mkdir ${DATAPATH}/tmp_agg_${PREFIX}_agg$4$5_${POSTFIX}_${CONFIG}
  mkdir -p ${DATAPATH}/snapshot/models/${PREFIX}_${stream}_${POSTFIX}_${CONFIG}
  mkdir -p ${DATAPATH}/snapshot/result/${PREFIX}_${stream}_${POSTFIX}_${CONFIG}

  cp ${DATAPATH}/checkpoints/ssd.pth ${DATAPATH}/tmp_agg_${PREFIX}_agg$4$5_${POSTFIX}_${CONFIG}/latest.pth
  cp ${DATAPATH}/checkpoints/ssd.pth ${DATAPATH}/snapshot/models/${PREFIX}_${stream}_${POSTFIX}_${CONFIG}/0.pth
done

for stream in $4 $5
do
  cp ${DATAPATH}/tmp_agg_${PREFIX}_agg$4$5_${POSTFIX}_${CONFIG}/latest.pth ${DATAPATH}/tmp_agg_${PREFIX}_${stream}_${POSTFIX}_${CONFIG}/latest.pth
  CUDA_VISIBLE_DEVICES=${DEVICE} python3 tools/model_test.py configs/custom/ssd_base.py ${DATAPATH}/tmp_agg_${PREFIX}_${stream}_${POSTFIX}_${CONFIG}/latest.pth --out ${DATAPATH}/tmp_agg_${PREFIX}_${stream}_${POSTFIX}_${CONFIG}/$(printf %02d 0).pkl -d ${PREFIX}_${stream}_test_0 -p ${DATAPATH}
done

for i in $(seq 1 1 $(expr ${SIZE} - 1))
do
  if [ $(expr ${i} % 3) -eq 0 ]; then
    CUDA_VISIBLE_DEVICES=${DEVICE} python3 tools/model_train.py configs/custom/ssd_fh.py --work-dir ${DATAPATH}/tmp_agg_${PREFIX}_agg$4$5_${POSTFIX}_${CONFIG}/ --train-dataset ${PREFIX}$4$5_1-10800_train_$(expr ${i} / 3 - 1) --no-validate --seed 0 --deterministic -p ${DATAPATH} --load-from ${DATAPATH}/checkpoints/ssd.pth
  fi
  for stream in $4 $5
  do
    CUDA_VISIBLE_DEVICES=${DEVICE} python3 tools/model_train.py configs/custom/ssd_fb.py --work-dir ${DATAPATH}/tmp_agg_${PREFIX}_${stream}_${POSTFIX}_${CONFIG}/ --train-dataset ${PREFIX}_${stream}_${POSTFIX}_train_$(expr ${i} - 1) --no-validate --load-from ${DATAPATH}/tmp_agg_${PREFIX}_agg$4$5_${POSTFIX}_${CONFIG}/latest.pth --seed 0 --deterministic -p ${DATAPATH}

    cp ${DATAPATH}/tmp_agg_${PREFIX}_${stream}_${POSTFIX}_${CONFIG}/latest.pth ${DATAPATH}/snapshot/models/${PREFIX}_${stream}_${POSTFIX}_${CONFIG}/${i}.pth

    CUDA_VISIBLE_DEVICES=${DEVICE} python3 tools/model_test.py configs/custom/ssd_base.py ${DATAPATH}/tmp_agg_${PREFIX}_${stream}_${POSTFIX}_${CONFIG}/latest.pth --out ${DATAPATH}/tmp_agg_${PREFIX}_${stream}_${POSTFIX}_${CONFIG}/$(printf %02d ${i}).pkl -d ${PREFIX}_${stream}_test_${i} -p ${DATAPATH}
  done
done

for stream in $4 $5
do
  for i in $(seq 0 1 $(expr ${SIZE} - 1))
  do
    cp ${DATAPATH}/tmp_agg_${PREFIX}_${stream}_${POSTFIX}_${CONFIG}/$(printf %02d ${i}).pkl ${DATAPATH}/snapshot/result/${PREFIX}_${stream}_${POSTFIX}_${CONFIG}/$(printf %02d ${i}).pkl
  done
done