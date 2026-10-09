"""Train or run the DQN agent in the Unity ML-Agents Banana environment.

Usage
=====
    python env.py 1    # train the agent (saves weights to artifact/checkpoint.pth)
    python env.py 0    # watch the trained agent collect bananas

Requires the Banana environment binary (Banana.app on macOS, or the
platform build from the README download link) plus the packages in
requirements.txt.
"""
import sys
import time
from collections import deque

import numpy as np
import torch

sys.path.append('util/')

from agent import Agent


def main(train):
    env = UnityEnvironment(file_name="Banana.app", base_port=64738, worker_id=3, seed=1)

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents in the environment
    print('Number of agents:', len(env_info.agents))

    # number of actions
    action_size = brain.vector_action_space_size
    print('Number of actions:', action_size)

    # examine the state space
    state = env_info.vector_observations[0]
    print('States look like:', state)
    state_size = len(state)
    print('States have length:', state_size)

    agent = Agent(state_size=state_size, action_size=action_size, seed=0)

    if train:
        train_agent(env, brain_name, agent)
    else:
        watch_agent(env, brain_name, agent)

    env.close()


def train_agent(env, brain_name, agent):
    """Train the agent with epsilon-greedy exploration until solved."""
    print("TRAINING")
    n_episodes = 1600
    max_t = 1000
    eps_start = 1.0
    eps_end = 0.01
    eps_decay = 0.998
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon

    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            env_ = env.step(action)[brain_name]
            next_state = env_.vector_observations[0]   # get the next state
            reward = env_.rewards[0]                   # get the reward
            done = env_.local_done[0]
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break

        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))

        if np.mean(scores_window) >= 17.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                         np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'artifact/checkpoint.pth')
            break

    from matplotlib.pylab import plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
    fig.savefig('artifact/scores.png', dpi=fig.dpi)

    torch.save(agent.qnetwork_local.state_dict(), 'artifact/checkpoint.pth')


def watch_agent(env, brain_name, agent):
    """Watch the trained agent act greedily (no learning)."""
    print("Smart Agent")
    agent.qnetwork_local.load_state_dict(torch.load('artifact/checkpoint.pth'))
    for i in range(3):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        for j in range(200):
            action = agent.act(state, eps=0.0)
            env_ = env.step(action)[brain_name]
            next_state = env_.vector_observations[0]   # get the next state
            reward = env_.rewards[0]                   # get the reward
            done = env_.local_done[0]
            state = next_state
            time.sleep(0.1)
            if done:
                break


if __name__ == "__main__":
    # The UnityEnvironment import is deferred so that `--help` style usage
    # errors are reported before the optional dependency is needed.
    try:
        from unityagents import UnityEnvironment
    except ImportError:
        print("The 'unityagents' package is required (pip install -r requirements.txt).")
        sys.exit(1)

    if len(sys.argv) != 2 or sys.argv[1] not in ("0", "1"):
        print("Usage: python env.py [0|1]   (0 = watch trained agent, 1 = train)")
        sys.exit(1)

    train_flag = int(sys.argv[1])
    print("Train : {}".format(train_flag))
    main(train_flag)
