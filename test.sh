PREFIX=$1
SIZE=$2
CUDA_VISIBLE_DEVICES=$3

python3 tools/merge_result.py -d tmp_${PREFIX} -c ${SIZE} -p result -f pkl
python3 tools/evaluate_from_file.py ${HOME}/urban/configs/ssd/ssdlite_mobilenetv2_scratch_600e_coco.py tmp_${PREFIX}/result.pkl -d ${PREFIX}
