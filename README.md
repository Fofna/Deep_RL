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

## Contributing

Feel free to contribute to this project by submitting pull requests or opening issues.




