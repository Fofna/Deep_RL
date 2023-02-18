# Deep Reinforcement learning from human preferences

Find the Article Link here : https://arxiv.org/abs/1706.03741

This repository contains Python scripts that implement various deep reinforcement learning algorithms, including Asynchronous Advantage Actor-Critic (A3C) and Deep Q-Network (DQN), and use human preferences as a form of reward to train agents to perform tasks. These scripts were used to generate data and conduct analysis for the research article titled "Deep Reinforcement Learning from Human Preferences".

The code is organized into several modules:

- `A2C.py`: implementation of the Advantage Actor-Critic (A2C) algorithm
- `Buffer.py`: implementation of the replay buffer used in the DQN algorithm
- `DQN.py`: implementation of the Deep Q-Network (DQN) algorithm
- `Q_table.py`: implementation of the Q-Table algorithm
- `Videos.py`: utility functions for generating videos of the agents playing games
- `a2c_network.py`: implementation of the neural network used in the A2C algorithm
- `deepA3C.py`: implementation of the Asynchronous Advantage Actor-Critic (A3C) algorithm with preference-based reward learning
- `main.py`: benchmark main to test reinforcement without human supervision
- `nnet.py`: implementation of the neural network used in the DQN algorithm
- `utils.py`: utility functions used across different modules

## Usage

To run the code, first install the required dependencies listed in `requirements.txt` by running:

pip install -r requirements.txt

Then, run the desired script. For example, to run the A2C algorithm, execute: A2C.py or deepA3C.py
The script will train the agent and save the learned model in a file named `a2c_model.h5`. The `Videos.py` module can be used to generate videos of the agent playing the game.

The `main.py` script can be used to run a benchmark of different reinforcement learning algorithms without human supervision. To run the benchmark, execute `main.py`


This will output the average rewards and standard deviation of each algorithm over 50 trials.

# Reproduction Results
![](results/pong.gif)
![](results/moving.gif)
![](result/enduro.gif)

The main objective of this reproduction project was to implement and reproduce the results of the paper "Deep Reinforcement Learning from Human Preferences" by Christiano et al. (2017). We successfully implemented the algorithms and conducted experiments on various environments, both synthetic and real, to reproduce the main results presented in the paper. Here we summarize our findings and compare them with the results reported in the original paper.

## Synthetic Environments
We started by reproducing the results on two simple synthetic environments, where we trained an agent to move a dot to the center of the screen and to play Pong, respectively. In both cases, we used synthetic preferences to guide the agent towards the optimal behavior. Our results were consistent with those presented in the original paper, demonstrating that the proposed algorithm is capable of learning from preferences, and that it achieves good performance in these simple environments.

## Real Environment
Next, we moved to a more complex environment, where we trained an agent to stay alongside other cars in the classic Atari game Enduro, using human preferences. We asked several people to play the game and select the trajectories that they found most appealing. We then used these preferences to train the agent. Our results showed that the agent was able to learn from human preferences and achieve a performance comparable to that of the human experts.

Overall, our reproduction project confirmed the main findings of the original paper and demonstrated the effectiveness of the proposed algorithm for learning from human preferences in both synthetic and real environments. Our implementation of the algorithm is available in this repository, and we encourage further research and experimentation in this area.


## Contributing

Feel free to contribute to this project by submitting pull requests or opening issues.




