# EANet

## This repository is an official implementation of the paper Extract and Adaptation Network for 3D Interacting Hand Mesh Recovery (ICCVW 2023)".
![alt text](model.png)

### Abstract
Understanding how two hands interact with each other is a key component of accurate 3D interacting hand mesh recovery. However, recent Transformer-based methods struggle to learn the interaction between two hands as they directly utilize two hand features as input tokens, which results in distant token problem. The distant token problem represents that input tokens are in heterogeneous spaces, leading Transformer to fail in capturing correlation between input tokens. Previous Transformer-based methods suffer from the problem especially when poses of two hands are very different as they project features from a backbone to separate left and right hand-dedicated features. We present EANet, extract-and-adaptation network, with EABlock, the main component of our network. Rather than directly utilizing two hand features as input tokens, our EABlock utilizes two complementary types of novel tokens, SimToken and JoinToken, as input tokens. Our two novel tokens are from a combination of separated two hand features; hence, it is much more robust to the distant token problem. Using
the two type of tokens, our EABlock effectively extracts interaction feature and adapts it to each hand. The proposed
EANet achieves the state-of-the-art performance on 3D interacting hands benchmarks.

#### Getting started

- Clone this repo.
```bash
git clone https://github.com/jkpark0825/EANet
cd EANet/main
```

- Install dependencies. (Python 3 + NVIDIA GPU + CUDA. Recommend to use Anaconda)
```bash
pip install -r requirements.txt
```

- Prepare the training and testing dataset. (https://mks0601.github.io/InterHand2.6M/)

#### Training
If you prefer not to train the model, you can simply obtain the pretrained model by downloading it from this link:
https://drive.google.com/file/d/14veExC7JG0jj1fXfIF0hz3dnxs7LqLXk/view?usp=sharing

Else, run the training code
```bash
python train.py --gpu 0,1,2,3
```

#### Testing
Note that implemented evaluation code is based on IntagHand (https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Interacting_Attention_Graph_for_Single_Image_Two-Hand_Reconstruction_CVPR_2022_paper.pdf).
```bash
python test.py --gpu 0,1,2,3 --test_epoch 29
```

#### demo
Prepare the cropped hand image (example.png), then run
```bash
python demo.py --gpu 0,1,2,3 --test_epoch 29 --input example_image1.png
```
![alt text](demo_example.png)
