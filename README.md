Usage
--

## Environment:
- anaconda
- python3.6
- pytorch
- matlab 2019

### Dataset

Two options: 
- [i-LIDS-Vid](http://www.eecs.qmul.ac.uk/~xiatian/iLIDS-VID/iLIDS-VID.tar)
- [prid](https://files.icg.tugraz.at/f/6ab7e8ce8f/?raw=1)

## Preprocess

1. modified line 34 in the matlab code `computeOpticalFlow.m` 
to your dataset path, then run the matlab code. This produces ground truth for training the MotionNet.
2. create anaconda environment by `conda create -n torch python==3.6 pytorch==1.3.0 ignite -c pytorch`.
3. activate conda environment and install required libraries: `pip install -r requirements.txt`.

## Training

1. Activate anaconda environment. `conda activate torch`.
2. (Optional) open a visdom service: `python -m visdom.server`.
3. Open another terminal/cmd, train the MotionNet sub model: 

   `python train_motionnet.py -l 1e-4 --saveFileName motionnet -dataset 0`.
   
4. Load the above pretrained motionnet model and train the main model:

   `python videoReid.py --train --usePredefinedSplit -l 1e-3 --saveFileName amoc -mp path/to/pretrained_motionnet_model.pth`.
   
## Testing

`python track_demo.py --source path/to/video_file --tracker KCF/GOTURN/CSRT -p path/to/amoc_weights.pth`