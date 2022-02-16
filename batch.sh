stream=1
for retrain in 000 020 040 060 080 100
do
  python3 tools/range_eval.py -r detrac_trace_${stream}_${retrain}-500_all_acc -d detrac_trace_${stream} -s True -n 20 > ${stream}-${retrain}.txt
done