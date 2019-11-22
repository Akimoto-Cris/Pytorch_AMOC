@echo off
call conda activate torch
cd /d D:\programs\PyTorch_AMOC
call python videoReid.py --usePredefinedSplit -l 1e-3 --saveFileName amoc_orth_clip -p trainedNets\amoc_orth_clip_model_802.pth

