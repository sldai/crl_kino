This package is used to solve the kinodynamic problem using RRT.  

## Installation

Now it is still in development, to install it for development:

Notice that if you have installed tianshou, please remove it or create another virtual env to install this package. Because we use tianshou for reinforcemenr learning, and change it a bit for our use.

```
conda create -n crl_kino python=3.6 
conda activate crl_kino
git clone https://github.com/sldai/crl_kino.git
cd crl_kino
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
pip install -e .
```

## Use

Test:

```
# kinodynamic RRT using dwa as steering function
python example/test.py --case rrt

# kinodynamic RRT using a trained policy as steering function
python example/test.py --case rl_rrt
```

Gif of kinodynamic rrt results will be saved in the current folder.


Train an agent:

```
python examples/rl_training.py 
```

## Results

RRT tree growing process
![path](/data/images/rrt_tree.gif)

The final course 
![path](/data/images/rrt_path.gif)