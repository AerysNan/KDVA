PREFIX='virat_1'

mkdir tmp_${PREFIX}
cp checkpoints/ssdlite_mobilenetv2_scratch_600e_coco_20210629_110627-974d9307.pth tmp_${PREFIX}/previous.pth
./tools/dist_test.sh /home/ubuntu/urban/configs/custom/ssd_${PREFIX}_0.py /home/ubuntu/urban/tmp_${PREFIX}/previous.pth 2 --eval bbox
for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
do
  python3 tools/train.py /home/ubuntu/urban/configs/custom/ssd_${PREFIX}_${i}.py --work-dir tmp_${PREFIX}/ --no-validate
  mv tmp_${PREFIX}/latest.pth tmp_${PREFIX}/previous.pth
  ./tools/dist_test.sh  /home/ubuntu/urban/configs/custom/ssd_${PREFIX}_${i}.py /home/ubuntu/urban/tmp_${PREFIX}/previous.pth 2 --eval bbox
done
