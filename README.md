# CRFormer: Complementary Reliability perspective Transformer (IEEE ACCESS)

This is a Pytorch implementation of [CRFormer: Complementary Reliability perspective Transformer for Automotive Components Reliability Prediction Based on Claim Data](https://ieeexplore.ieee.org/document/9863836).

The overall architecture of CRFormer is as below:

<p align="center">
	<img src="./img/model1.png" alt="CRFormer" width="60%" height="60%"/>
</p>

# Installation & Enviornment

The OS, python and pytorch version needs as below:
- Windows
- Linux 
- python >= 3.7.4
- pytorch >= 1.7.1

# How to use

## 1. Prepare dataset

We use VoiceBank-DEMAND (Valentini) dataset consisting of 28 speakers for training MANNER. 

- The dataset can be downloaded [here](https://datashare.ed.ac.uk/handle/10283/2791).
- We use [282,287] speakers as validation set.

## 2. Downsample

The sample rate of the dataset is 48kHz.

For a fair comparison, we downsample the audio files from 48kHz to 16kHz.

- To downsample the audio, run the following code and edit the directories.

```
python downsampling.py
```
  
- In the ```downsampleing.py``` script, you should change the contents as follows.
  
```
downsample_rate  = 16000
clean_train_path = 'The original clean trainset path'
noisy_train_path = 'The original noisy trainset path'
clean_test_path  = 'The original clean testset path'
noisy_test_path  = 'The original noisy testset path'
resample_path    = 'Resampled path'
```
  
## 3. Make data path files

We make json file consisting of the audio path for loading data efficiently. Train (clean, noisy) and 
Test (clean, noisy): four json files need to be generated for training. 

The json files will be generated in ```./data_path```.

Notice that the data is downsampled.

- To make json file, run the following code and edit the directories.

```
python make_datapath.py
```

- In the ```make_datapath.py```, you should change the contents as follows.

```
clean_train_path = 'The resampled clean trainset path'
noisy_train_path = 'The resampled noisy trainset path'
clean_test_path  = 'The resampled clean testset path'
noisy_test_path  = 'The resampled noisy testset path'
```

# How to use

## 1. Train

### Training with default settings

You can train MANNER with the default setting by running the following code.

```
python main.py train --aug True --aug_type tempo
```

### Training with other arguments
If you want to edit model settings, you can run the following code with other arguments. 

In ```config.py```, you can find other arguments, such as batch size, epoch, and so on.

```
python main.py train --hidden 60 --depth 4 --growth 2 --kernel_size 8 --stride 4 --segment_len 64 --aug True --aug_type tempo

MANNER arguments:
  --in_channels : initial in channel size (default:1)
  --out_channels: initial out channel size (default:1)
  --hidden      : channel size to expand (default:60)
  --depth       : number of layers for encoder and decoder (default:4)
  --kernel_size : kernel size for UP/DOWN conv (default:8)
  --stride      : stride for UP/DOWN conv (default:4)
  --growth      : channel expansion ration (default:2)
  --head        : number of head for global attention (default:1)
  --segment_len : chunk size for overlapped chunking in a dual-path processing (default:64)
  
Setting arguments:
  --sample_rate: sample_rate (default:16000)
  --segment    : segment the audio signal with seconds (default:4)
  --set_stride : Overlapped seconds when segment the signal (default:1)
  
Augmentation arguments:
  --aug     : True/False 
  --aug_type: augmentation type (tempo, speed, shift available. only shift available on Windows.)
```

### Training with logging

The logs are uploaded on [neptune.ai](https://neptune.ai/)
```
python main.py train --logging True --logging_cut -1

Logging arguments:
  --logging    : True/False
  --logging_cut: log after epochs when the epoch is bigger than logging_cut
```

## 2. evaluation

After training, you can evaluate the model in terms of PESQ and STOI by running the code below.
You need to keep the model arguments in the training phase.
```
python main.py test --save_enhanced True --enhanced_path []

evaluation arguments:
  --save_enhanced: saving enhanced audio file
  --enhanced_path: enhanced file directory
```

If you want to evaluate with all measures (PESQ, STOI, CSIG, CBAK, COVL), run the following code.
```
python eval_measure.py

clean_path    = 'test clean path'
enhanced_path = 'enhanced path'
```


# Experimental Results

We provide visualizations of result samples. They are predicted by CRFormer, CRFormer-F, and other benchmark models.
(a) and (b) indicate prediction results for time (days) and mileage, respectively.

<p align="center">
	<img src="./img/results.png" alt="results" width="80%" height="80%"/>
</p>




## Citation

```
@article{park2022crformer,
  title={CRFormer: Complementary Reliability Perspective Transformer for Automotive Components Reliability Prediction Based on Claim Data},
  author={Park, Hyun Joon and Kim, Taehyeong and Kim, Young Seok and Min, Jinhong and Sung, Ki Woo and Han, Sung Won},
  journal={IEEE Access},
  volume={10},
  pages={88457--88468},
  year={2022},
  publisher={IEEE}
}
```

## License

This repository is released under the MIT license.
