# Banana Navigation with a Dueling Double DQN

A deep Q-network agent in PyTorch that learns to collect yellow bananas and avoid blue ones in Unity ML-Agents, using a dueling head, Double DQN targets and experience replay. Built for the Udacity Deep RL Nanodegree.

## What's inside

- `env.py` trains or runs the agent
- `util/` holds the agent, Q-network and replay buffer
- `Navigation.ipynb` is the notebook version

## How to run

```bash
pip install -r requirements.txt
python env.py 1   # train
python env.py 0   # watch the trained agent
```

Use Python 3.6, which the `unityagents` package needs. `env.py` opens the included macOS build `Banana.app`; on other systems, point it at the matching Banana build.

## Related tutorial

- [Deep Q-Networks](https://raminaeye.github.io/dqn.html)
