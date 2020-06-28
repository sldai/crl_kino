This package is used to solve the kinodynamic problem using RRT.  

## Installation

Now it is still in development, to install it for development:

Notice that if you have installed tianshou, please remove it or create another virtual env to install this package. Because we use tianshou for reinforcemenr learning, and change it a bit for our use.

```
conda create -n crl_kino python=3.7 
conda activate crl_kino
# install pytorch
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
pip install -e .
```

## Use


```
# train a local planner
python example/rl_training.py

# rrt planning with the trained rl local planner
python example/show.py --case rrt

# rrtstar planning with the trained rl local planner
python example/test.py --case rl_rrt
```

Trained models can be found in the arclab server `~/shilong/crl_kino/log`

## Results

RRT tree
![path](/data/images/rrt_tree.png)

RRT* tree
![path](/data/images/rrtstar_tree.png)