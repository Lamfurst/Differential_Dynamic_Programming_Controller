# DESCRIPTION
This repo implements following: 

- Differential Dynamic Programming (DDP) controller using `autograd` package. 
- MPPI controller 
- Implement dynamics for Cartpole 
- Implement dynamics for inverted double-pendulum on a cart
- Achieve cartpole control using DDP
- Achieve swing-up control using DDP
- Achieve swing-up control using MPPI

# Demos

## Cartpole Control (DDP)
<img src="https://github.com/Lamfurst/Differential_Dynamic_Programming_Controller/blob/main/visual_result/cartpole_best.gif" width="300" height="auto" />

## Swing-up Control (DDP)
<img src="https://github.com/Lamfurst/Differential_Dynamic_Programming_Controller/blob/main/visual_result/ddp_double_pendulum.gif" width="300" height="auto" />

## Swing-up Control (MPPI)
<img src="https://github.com/Lamfurst/Differential_Dynamic_Programming_Controller/blob/main/visual_result/mppi_double_pendulum.gif" width="300" height="auto" />


# Getting Started

## Install require packages

```bash
$ ./install.sh
```

## Run swing-up demo for inverted double-pendulum on a cart

```bash
$ python demo.py
```

# Other examples

The codes to achieve controls for other scenarios are included in `Other_Examples` folder. You can run `test.ipynb` to get the demo results shown in `Demos` section above. 