# Capsule Network #
[![License](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](LICENSE)

#### PyTorch implementation of the following paper:
* [_Dynamic Routing Between Capsules_](https://arxiv.org/abs/1710.09829) by Sara Sabour, Nicholas Frosst and Geoffrey Hinton

### Run the experiment
* For details, run `python main.py -e 100 -bs 32 --lr_decay 0.99 --data_path train7`

______

### Requirements:
* PyTorch (http://www.pytorch.org)
* NumPy (http://www.numpy.org/)
* GPU

[//]: # (### Loss function hyper-parameters (see [loss.py](loss.py)):)
[//]: # (* Lambda for Margin Loss = 0.5)
[//]: # (* Scaling factor for reconstruction loss = 0.0005)
