@echo off
call conda activate torch
cd /d D:\programs\PyTorch_AMOC
call python videoReid.py --train --usePredefinedSplit -l 1e-3 --saveFileName amoc_orth_clip -mp trainedNets\motionnet_model_98.pth