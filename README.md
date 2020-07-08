# Attack by Identifying Effective Features

This repository contains the code for [Generate High-Resolution Adversarial Samples by Identifying Effective Features](https://arxiv.org/abs/2001.07631) (Arxiv).


## Method

We propose Attack by Identifying Effective Features (AIEF), which learns different weights for features to attack.


## Environment

This code is implemented in Tensorflow, and we have tested the code under the following environment settings:

- python = 3.7.5
- tensorflow-gpu = 1.13.2
- models/facenet/model-20180402-114759.ckpt-275.data-00000-of-00001 [download](https://github.com/davidsandberg/facenet)
- models/stylegan/karras2019stylegan-celebahq-1024x1024.pkl [download](https://github.com/NVlabs/stylegan)

## Run the code

The given code shows how to implement AIEF on StyleGAN (trained on CelebA-HQ). 

We give a demo images in the folder'./img' which is chosen from  CelebA-HQ, and a trained StyleGAN in'./models'. To implement our method, just run the code below.

 ```
python main.py
 ```

## Results

<img src="./img/1.png" width=400px><img src="./result/adv1.png" width=400px>

<img src="./img/2.png" width=400px><img src="./result/adv2.png" width=400px>

