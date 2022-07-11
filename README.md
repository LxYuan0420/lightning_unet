PyTorch Lightning Implementation of UNET model for Semantic Segmentation
---

- Original paper: [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- Dataset: [Carvana Image Masking Challenge from Kaggle](https://www.google.com/search?q=carvana+dataset+kaggle&oq=carv&aqs=chrome.0.69i59j69i57j69i60l3.1204j0j1&sourceid=chrome&ie=UTF-8)

#### 0. Repository Structure
```
.
├── data/
│   ├── train/
│   │   ├── 00087a6bd4dc_01.jpg
│   │   ├── 00087a6bd4dc_02.jpg
│   │   ├── 00087a6bd4dc_03.jpg
│   │   ├── 00087a6bd4dc_04.jpg
│   │   ├── 00087a6bd4dc_05.jpg
│   │   └── ...
│   └── train_masks/
│       ├── 00087a6bd4dc_01_mask.gif
│       ├── 00087a6bd4dc_02_mask.gif
│       ├── 00087a6bd4dc_03_mask.gif
│       ├── 00087a6bd4dc_04_mask.gif
│       ├── 00087a6bd4dc_05_mask.gif
│       └── ...
├── README.md
├── dataset.py
├── model.py
├── requirements.txt
├── train.py
└── kaggle.json
└── train.py

```

#### 1. Getting started
```
# clone repository
$ git clone https://github.com/LxYuan0420/lightning_unet.git
$ cd lightning_unet/

# activate virtual env
$ python -m venv v
$ source env/bin/activate

# install libaries
(env)$ pip install -r requirements.txt

# get kaggle api token from your kaggle account
(env)$ mkdir -p ~/.kaggle/ && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

# accept compentition rules and download dataset
(env)$ kaggle competitions download -c carvana-image-masking-challenge

# unzip and remove other files to make sure your data/ dir is
# the same as the expected repository structure.
```

#### 2. Start training
```
# optional: modify hyperparameters
(env)$ vim train.py

(env)$ python train.py

...
Epoch 8: 100%|██████████| 319/319 [01:01<00:00,  5.21it/s, loss=0.0198, v_num=2, train_loss_step=0.0183, val_loss=0.310, val_acc=0.914, val_f1=0.832, train_loss_epoch=0.0193]

(env)$ ls unet_models/
'unet_implementation-epoch-epoch=06-val_loss-val_loss=0.02-val_acc-val_acc=0.99-val_f1-val_f1=0.98.ckpt'
```
