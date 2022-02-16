stream=$1

cp snapshot/gt/detrac_trace_${stream}_m.pkl snapshot/merge/detrac_trace_${stream}_000-500_all_acc.pkl
python3 tools/expand_result.py -r snapshot/gt/detrac_trace_${stream}_m.pkl -d snapshot/result/detrac_trace_${stream}_000-500_all_acc
python3 tools/range_eval.py -r detrac_trace_${stream}_000-500_all_acc -d detrac_trace_${stream}_020-500 -s True -n 20