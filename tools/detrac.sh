#!/bin/bash
# files=(MVI_39761 MVI_39771 MVI_39801 MVI_39811 MVI_39821 MVI_39861 MVI_40141 MVI_40152 MVI_63544)
# files=(MVI_20011 MVI_20012 MVI_20032 MVI_20033 MVI_20034 MVI_20035 MVI_20051 MVI_20052 MVI_20061 MVI_20062 MVI_20063 MVI_20064 MVI_20065 MVI_39781 MVI_39851 MVI_40131 MVI_40161 MVI_40162 MVI_40171 MVI_40172 MVI_40181 MVI_63521 MVI_63525)
# files=(MVI_40191 MVI_40192 MVI_40201 MVI_40204 MVI_40211 MVI_40212 MVI_40213 MVI_40241 MVI_40243 MVI_40244 MVI_41063 MVI_41073 MVI_63552 MVI_63553 MVI_63554 MVI_63561 MVI_63562 MVI_63563)
files=(MVI_40701 MVI_40851 MVI_40852 MVI_40853 MVI_40854 MVI_40855 MVI_40891 MVI_40892 MVI_40901 MVI_40902 MVI_40903 MVI_40904 MVI_40905)
for ((i=0;i<${#files[@]};i++))
do
  cp -r ~/DETRAC/videos/${files[i]} data/_detrac_trace_$i
  python3 tools/dataset_converters/images2coco.py ./data/_detrac_trace_$i/ ./classes.dat _detrac_trace_$i.base.json
  python3 tools/dataset_converters/xml_to_annotation.py -p ~/DETRAC/annotations/${files[i]}.xml -d _detrac_trace_$i
  echo _detrac_trace_$i >> tmp.txt
done

python3 tools/merge_traces.py -o merge_trace_4 -i tmp.txt
rm tmp.txt
rm -rf data/_detrac*
rm data/annotations/_detrac*