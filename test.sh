PREFIX='virat_1'

for i in $(seq 0 1 40)
do
  ./tools/dist_test.sh /home/ubuntu/urban/configs/custom/ssd_${PREFIX}_${i}.py /home/ubuntu/urban/checkpoints/ssdlite_mobilenetv2_scratch_600e_coco_20210629_110627-974d9307.pth 2 --eval bbox
done
