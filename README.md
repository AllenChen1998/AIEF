# Attack by Identifying Effective Features

This repository contains the code for [Generate High-Resolution Adversarial Samples by Identifying Effective Features](https://arxiv.org/abs/2001.07631) (Arxiv).


## Method

We propose Attack by Identifying Effective Features (AIEF), which learns different weights for features to attack.


## Requisite

This code is implemented in Tensorflow, and we have tested the code under the following environment settings:

- python = 3.7.5
- tensorflow-gpu = 1.13.2

## Run the code

The given code shows how to implement AIEF on StyleGAN (trained on CelebA-HQ). 

We give a demo images in the folder'./img' which is chosen from  CelebA-HQ, and a trained StyleGAN in'./models'. To implement our method, just run the code below.

 ```
python main.py
 ```

## Results

<img src="./img/1.png" width=400px ><img src="./result/adv1.png" width=400px align="center">

<img src="./img/2.png" width=400px><img src="./result/adv2.png" width=400px>

