#!/bin/bash

output_dir="/home/gucheol/Workspace/CreateDataSet/TextRecognitionDataGenerator/out/eng"
dictionary="/home/gucheol/data/hc_recog_data/train/random_eng_dict.txt"
font_dir="/home/gucheol/Workspace/CreateDataSet/TextRecognitionDataGenerator/trdg/fonts/kr"
#generate_num = countx12x36
count="400"

# python ./trdg/run.py --output_dir ${output_dir}/h40_bl0_b0_cs0 -l kr -bl 0 -b 0 -c ${count} -w 1 -cs 0 -f 40 -tc '#000000,#b5b5b5' -dt $dictionary -fd $font_dir

#img_h
for ((i=30; i<=64; i++))
do
    #background
    for ((j=0; j<=2; j++))
    do
        #char_spacing
        for ((k=0; k<=3; k++))
        do
            # blur(bl) 0~1, background(b) 0~2, char_spacing(cs) 0~3, 5color(grey)
            python ./trdg/run.py --output_dir ${output_dir}/h${i}_bl0_b${j}_cs${k} -l kr -bl 0 -b ${j} -c ${count} -w 1 -cs ${k} -f ${i} -tc '#000000,#b5b5b5' -dt $dictionary -fd $font_dir
            python ./trdg/run.py --output_dir ${output_dir}/h${i}_bl1_b${j}_cs${k} -l kr -bl 1 -b ${j} -c ${count} -w 1 -cs ${k} -f ${i} -tc '#000000,#b5b5b5' -dt $dictionary -fd $font_dir 
        done
    done
done

