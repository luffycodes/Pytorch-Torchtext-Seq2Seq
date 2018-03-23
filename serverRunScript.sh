#!/bin/bash

cd /root/pythonProjects/

rm -rf /root/pythonProjects/Pytorch-Torchtext-Seq2Seq
rm -rf /dataroot/Pytorch-Torchtext-Seq2Seq-Data/model_results/tfEvents
rm /dataroot/Pytorch-Torchtext-Seq2Seq-Data/model_results/console.log

git clone https://github.com/luffycodes/Pytorch-Torchtext-Seq2Seq.git --branch nodecoder --single-branch

cd Pytorch-Torchtext-Seq2Seq

nohup python main.py &
tail -f nohup.out