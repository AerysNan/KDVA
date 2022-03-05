# for retrain in 020 060 100 040 080
# do
#   for stream in $(seq 7 1 12)
#   do
#     ./train_test_na.sh sub_${stream} 12 1 ${retrain} na > ${stream}-${retrain}-na.txt 2> ${stream}-${retrain}-na-log.txt
#   done
# done

for retrain in 020 060 100 040 080
do
  for stream in $(seq 7 1 12)
  do
    ./train_test_na_val.sh sub_${stream} 12 1 ${retrain} esna > ${stream}-${retrain}-esna.txt 2> ${stream}-${retrain}-esna-log.txt
  done
done

./tools/scripts/validation.sh 7 12 1 esna