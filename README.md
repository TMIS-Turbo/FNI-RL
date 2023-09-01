# FNI-RL
This repo is the implementation of "**Fear-Neuro-Inspired Reinforcement Learning for Safe Autonomous Driving**".  

The code will be released soon.

## Demonstration
The video demonstration of our work can be found:
[Please click to watch the video](https://www.bilibili.com/video/BV1Mk4y157Da/?spm_id_from=333.337.search-card.all.click&vd_source=71620ac61fcf7851589c019bff140478).

In all demonstrations, the red-colored car represents the FNI-RL-driven autonomous vehicle.

###  1. Unprotected left turn at an unsignalized intersection with oncoming traffic
<img src="gif/env-(a).gif" alt="Scenario (a)" width="300" height="300">

###  2. Right turn at an unsignalized intersection with crossing traffic
<img src="gif/env-(b).gif" alt="Scenario (b)" width="300" height="300">

###  3. Unprotected left turn at an unsignalized intersection with mixed traffic flows
<img src="gif/env-(c).gif" alt="Scenario (c)" width="300" height="300">

###  4. Crossing negotiation at an unsignalized intersection with mixed traffic flows
<img src="gif/env-(d).gif" alt="Scenario (d)" width="300" height="300">

###  5. Long-term goal-driven navigation with mixed traffic flows
<img src="gif/env-(e)-1.gif" alt="Scenario (e1)" width="500" height="300">
<img src="gif/env-(e)-2.gif" alt="Scenario (e2)" width="500" height="300">

## Installation
This repo is developed using Python 3.7 and PyTorch 1.3.1+CPU in Ubuntu 16.04. 

We believe that our code can also run on other systems with different versions of Python or PyTorch, but we have not verified it.

It should be noted that, based on development experience, if using a higher version of PyTorch to run code developed based on a lower version of PyTorch, it may be necessary to adjust the order of the loss function and its updated model parameter modules in the training algorithm.

We utilize the proposed FNI-RL approach to train the autonomous driving agent in the popular [Simulation of Urban Mobility](https://eclipse.dev/sumo/) (SUMO, Version 1.2.0) platform.


