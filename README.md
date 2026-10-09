# Banana Navigation (DQN)

Ramin Anushiravani

A value-based deep reinforcement learning agent that learns to navigate the Unity
ML-Agents **Banana Collector** environment and collect yellow bananas while avoiding
blue ones.

### The environment

The agent moves through a large square field with a single goal: collect as many
yellow bananas as possible while avoiding blue bananas.

- **State:** a 37-dimensional vector describing the agent's velocity and ray-based
  perception of nearby objects.
- **Actions:** 4 discrete moves (forward, backward, turn left, turn right).
- **Rewards:** +1 for a yellow banana, -1 for a blue banana.
- **Solved:** an average score of 13.0 or higher over 100 consecutive episodes
  (the training script in this repo uses a stricter early-stop bar of 17.0).

### The agent

The agent is a **Deep Q-Network** with three value-based ingredients:

- **Dueling Q-network** (`util/qn.py`): a shared feature extractor feeds a value
  stream V(s) and an advantage stream A(s, a), combined as
  Q(s, a) = V(s) + A(s, a) - mean(A(s, a)).
- **Double DQN targets** (`util/agent.py`): the greedy next action is selected with
  the local network and evaluated with the target network, reducing Q-value
  overestimation.
- **Soft target updates** with tau = 1e-3, plus experience replay
  (`util/replay.py`, buffer size 100,000, batch size 64).

Training uses epsilon-greedy exploration: epsilon starts at 1.0, decays by 0.998 per
episode, and bottoms out at 0.01. Other hyperparameters: gamma = 0.99, learning
rate = 5e-4, network update every 4 steps.

### Repo structure

- `env.py`: train the agent (`python env.py 1`) or watch the trained agent
  (`python env.py 0`). Also split into `train_agent` and `watch_agent`.
- `util/agent.py`: the DQN agent (action selection, learning step, soft updates).
- `util/qn.py`: the dueling Q-network.
- `util/replay.py`: the experience replay buffer.
- `Navigation.ipynb`: the full training notebook (exploration, training, watching),
  mirroring `env.py`.
- `Navigation_Pixels.ipynb`: template for the pixel-observation variant (raw visual
  input instead of the state vector); training scaffold only.
- `artifact/checkpoint.pth`: trained network weights.
- `artifact/scores.png`: per-episode scores from the training run.
- `artifact/smart_banana.mov`: short video of the trained agent collecting bananas.
- `report.html`: exported report of the original notebook.
- `Banana.app`: the Unity environment build for macOS (large binary, committed).

### How to run

Training requires the Unity ML-Agents Banana environment. `Banana.app` is included
for macOS; for other platforms download the matching build (the original project
also shipped `Banana_Linux.zip`, `Banana.app.zip`, and `Banana_Windows_x86_64.zip`).

```bash
pip install -r requirements.txt

python env.py 1   # train the agent, saves weights to artifact/checkpoint.pth
python env.py 0   # watch the trained agent collect bananas
```

If you have trouble launching the environment from the notebook, use the `env.py`
script instead; both contain the same training loop.

### Results

The trained agent reached the early-stop bar: average score >= 17.0 over the last
100 episodes. The score plot from the training run:

![scores](artifact/scores.png)

### Future improvements

The reward curve is noisy, so a few directions could stabilize and improve training:

- **Prioritized experience replay**, to sample informative transitions more often.
- **Rainbow DQN** (combining prioritized replay, n-step targets, distributional
  values, noisy nets) or a deeper Q-network.
- Longer training runs or more collected experience per update.
