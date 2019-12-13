# muzero-pytorch
Pytorch Implementation of MuZero : "[Mastering Atari , Go, Chess and Shogi by Planning with a Learned Model](https://arxiv.org/pdf/1911.08265.pdf)"  based on [pseudo-code](https://arxiv.org/src/1911.08265v1/anc/pseudocode.py) provided by the authors

_Note: This implementation has just been tested on CartPole-v1 and would required modifications(`in config folder`) for other environments_

## Installation
```bash
cd muzero-pytorch
pip install -r requirements.txt
```
## Usage:
* Train: ```python main.py --env CartPole-v1 --case classic_control --opr train --force```
* Test: ```python main.py --env CartPole-v1 --case classic_control --opr test```