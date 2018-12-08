#!/bin/bash 

for batch_size in {64,128,256,512} 
do   
    for l2 in {0.5,0.2,0.1,0.05,0.02,0.01}
    do
        for from_num in {60,0,5,10,15,20,25,30,35,40}
        do
            python3 keras_lstm.py 1 --batch_size ${batch_size} --l2 ${l2} --from_num ${from_num} --use_dropout False --verbose 1 --num_steps 100 --epoch_num 10
        done
    done
done  
