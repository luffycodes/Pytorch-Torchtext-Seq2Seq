#!/bin/bash

rm -rf /dataroot/Pytorch-Torchtext-Seq2Seq-Data/model_results/tfEvents
rm /dataroot/Pytorch-Torchtext-Seq2Seq-Data/model_results/console.log
rm nohup.out
git pull origin nodecoder
nohup python main.py &
tail -f nohup.out