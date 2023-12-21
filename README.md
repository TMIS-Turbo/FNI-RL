# FNI-RL
This repo is the implementation of our research "**[Fear-Neuro-Inspired Reinforcement Learning for Safe Autonomous Driving](https://www.researchgate.net/publication/374522737_Fear-Neuro-Inspired_Reinforcement_Learning_for_Safe_Autonomous_Driving)**".  

## Introduction
### Schematic of the Proposed FNI-RL Framework for Safe Autonomous Driving
<img src="images/framework.jpg" alt="ENV" width="500" height="300">
In recent years, many neuroscientists have argued that, the specific fear nervous system (i.e., amygdala) in the brain plays a central role, which can predict dangers and elicit defensive behavioral responses against threats and harms; this is crucial for survival in and adaptation to potential risky environments. Additionally, some studies in neuroscience and psychology have highlighted the necessity of actively forecasting hazards or contingencies via world models to ensure the survival of organisms. Motivated by the above insights, we devised a brain-like machine intelligence paradigm by introducing an adversarial imagination mechanism into a constrained reinforcement learning framework, allowing autonomous vehicles to acquire a sense of fear, thereby enhancing or ensuring safety.

## Demonstration
The video demonstration of our work can be found:
[Please click to watch the video](https://www.bilibili.com/video/BV1E34y1T73M/?spm_id_from=333.337.search-card.all.click&vd_source=71620ac61fcf7851589c019bff140478).

In all demonstrations, the red-colored car represents the FNI-RL-driven autonomous vehicle.

###  1. Unprotected left turn at an unsignalized intersection with oncoming traffic
<img src="images/env-(a).gif" alt="Scenario (a)" width="300" height="300">

###  2. Right turn at an unsignalized intersection with crossing traffic
<img src="images/env-(b).gif" alt="Scenario (b)" width="300" height="300">

###  3. Unprotected left turn at an unsignalized intersection with mixed traffic flows
<img src="images/env-(c).gif" alt="Scenario (c)" width="300" height="300">

###  4. Crossing negotiation at an unsignalized intersection with mixed traffic flows
<img src="images/env-(d).gif" alt="Scenario (d)" width="300" height="300">

###  5. Long-term goal-driven navigation with mixed traffic flows
<img src="images/env-(e)-1.gif" alt="Scenario (e1)" width="500" height="300">
<img src="images/env-(e)-2.gif" alt="Scenario (e2)" width="500" height="300">

## Installation
This repo is developed using Python 3.7 and PyTorch 1.3.1+CPU in Ubuntu 16.04. 

We utilize the proposed FNI-RL approach to train the autonomous driving agent in the popular [Simulation of Urban Mobility](https://eclipse.dev/sumo/) (SUMO, Version 1.2.0) platform.

We believe that our code can also run on other operating systems with different versions of Python, PyTorch and SUMO, but we have not verified it.

The required packages can be installed using

	pip install -r requirements.txt


