Kinodynamic RRT using compositional reinforcement learning

## Installation

Now it is still in development, to install it for development:

```
conda create -n crl_kino python=3.6 
conda activate crl_kino
git clone
cd MPNet_RL
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
pip install -e .
```

## Use

Test:

```
python example/test.py --case rrt
```

Gif of kinodynamic rrt results will be saved in the current folder.