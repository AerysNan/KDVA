# for retrain in 020 060 100 040 080
# do
#   for stream in $(seq 1 1 6)
#   do
#     ./train_test_na.sh sub_${stream} 12 0 ${retrain} na > ${stream}-${retrain}-na.txt 2> ${stream}-${retrain}-na-log.txt
#   done
# done

for retrain in 020 060 100 040 080
do
  for stream in $(seq 1 1 6)
  do
    ./train_test_na_val.sh sub_${stream} 12 0 ${retrain} esna > ${stream}-${retrain}-esna.txt 2> ${stream}-${retrain}-esna-log.txt
  done
done

./tools/scripts/validation.sh 1 6 0 esna

# for retrain in 000 020 040 060 080 100
# do
#   for stream in $(seq $1 1 $2)
#   do
#     python3 tools/range_eval.py -r sub_${stream}_${retrain}_short -d sub_${stream} -s False -n 12 > ${stream}-${retrain}.txt
#   done
# done
# for retrain in 020 040 060 080 100
# do
#   for epoch in $(seq 0 1 11)
#   do
#     echo sub_$1_${retrain}_train_$epoch >> tmp.txt
#     echo sub_$2_${retrain}_train_$epoch >> tmp.txt
#     python3 tools/merge_traces.py -i tmp.txt -o merge$1$2_${retrain}_train_$epoch --gt False
#     rm -rf tmp.txt
#   done
# done

