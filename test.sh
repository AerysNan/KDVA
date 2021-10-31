PREFIX='virat_1'

for i in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
do
  ./tools/dist_test.sh /home/ubuntu/urban/configs/custom/ssd_${PREFIX}_${i}.py /home/ubuntu/urban/checkpoints/ssdlite_mobilenetv2_scratch_600e_coco_20210629_110627-974d9307.pth 2 --eval bbox
done
