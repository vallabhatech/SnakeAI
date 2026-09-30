# 🐍 Snake AI

A Deep Q-Learning agent that learns to play the classic Snake game using **PyTorch**, **Pygame**, and experience replay.

The project is intentionally compact: the agent observes an 11-feature representation of the game state, chooses one of three relative actions, receives a reward, and updates its neural network from experience.

[![Python](https://img.shields.io/badge/Python-3.7%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Pygame](https://img.shields.io/badge/Pygame-Game%20Environment-00A000)](https://www.pygame.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## ✨ What this project does

Snake AI trains an agent through reinforcement learning rather than hard-coded game rules.

The training loop repeatedly:

1. Reads the current game state.
2. Chooses an action with an epsilon-greedy policy.
3. Executes that action in the Snake environment.
4. Receives a reward.
5. Learns immediately from the transition.
6. Stores the transition in replay memory.
7. Trains on replayed experiences after each episode.
8. Saves the model whenever a new record score is reached.

## 🧠 Architecture

```text
┌──────────────────────┐
│   Snake Environment  │
│     snake_game.py    │
└──────────┬───────────┘
           │ state / reward
           ▼
┌──────────────────────┐
│        Agent         │
│      agent.py        │
│  ε-greedy + replay   │
└──────────┬───────────┘
           │ state
           ▼
┌──────────────────────┐
│      Q-Network       │
│       model.py       │
│  11 → 256 → 3        │
└──────────┬───────────┘
           │ Q-values
           └──────────────► action
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for a deeper explanation of the data flow and learning process.

## 🎯 State representation

The network receives **11 binary features**:

| Group | Features |
|---|---|
| Danger | Straight, right, left |
| Direction | Left, right, up, down |
| Food | Left, right, up, down |

This compact representation avoids feeding the entire board into the network and keeps the model lightweight.

## 🎮 Action space

Actions are relative to the snake's current direction:

```text
[1, 0, 0] → Straight
[0, 1, 0] → Right
[0, 0, 1] → Left
```

Using relative actions prevents direct 180° turns and makes the policy independent of the snake's absolute orientation.

## 🧮 Learning setup

The current implementation uses:

- **Algorithm:** Deep Q-Learning
- **Network:** Fully connected neural network
- **Architecture:** 11 inputs → 256 hidden units → 3 outputs
- **Activation:** ReLU
- **Optimizer:** Adam
- **Learning rate:** 0.001
- **Discount factor (γ):** 0.9
- **Replay memory:** 100,000 transitions
- **Replay batch size:** 1,000
- **Exploration:** epsilon-greedy
- **Reward for food:** +10
- **Reward for collision:** -10
- **Reward for normal movement:** 0
- **Game grid:** 640 × 480 pixels
- **Cell size:** 20 pixels
- **Game speed:** 100 FPS

CUDA is used automatically when PyTorch detects a compatible GPU; otherwise training falls back to CPU.

## 📁 Project structure

```text
SnakeAI/
├── agent.py                 # RL agent, state encoding, replay and training loop
├── model.py                 # Q-network and Q-learning trainer
├── snake_game.py            # Pygame environment and game mechanics
├── helper.py                # Training-score visualization
├── model/
│   └── model.pth            # Saved model weights when present
├── docs/
│   └── ARCHITECTURE.md      # Architecture and learning documentation
├── .vscode/                 # Editor configuration
├── arial.ttf                # Font used by the game UI
├── thumbnail.png            # Demo thumbnail
├── notes.txt                # Original development notes
├── requirements.txt         # Python dependencies
├── CHANGELOG.md             # Project change history
├── CONTRIBUTING.md          # Contribution guidelines
├── LICENSE                  # MIT license
└── README.md                # Project documentation
```

## 🚀 Getting started

### Prerequisites

- Python 3.7 or newer
- pip
- Optional: a CUDA-capable GPU with a compatible PyTorch installation

### 1. Clone

```bash
git clone https://github.com/vallabhatech/SnakeAI.git
cd SnakeAI
```

### 2. Create a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Linux/macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Train the agent

```bash
python agent.py
```

A Pygame window will display the game while Matplotlib tracks training scores.

Training continues until the game window is closed.

## 💾 Model persistence

When the agent achieves a new record score, its network weights are saved to:

```text
model/model.pth
```

The saved file contains the PyTorch `state_dict`. To load it in another script:

```python
import torch
from model import Linear_QNet

model = Linear_QNet(11, 256, 3)
model.load_state_dict(torch.load("model/model.pth", map_location="cpu"))
model.eval()
```

For production or repeatable experiments, pinning dependency versions and keeping training checkpoints with experiment metadata is recommended.

## 📊 Training and evaluation

The current training program is designed primarily as a learning/demo implementation.

It reports:

- Current game number
- Current score
- Best record
- Running mean score
- Live score visualization

The repository does **not** currently provide a statistically rigorous benchmark suite, fixed-seed evaluation harness, or experiment tracking system. Reported scores should therefore be treated as observations from training runs rather than reproducible benchmark results.

## 🔍 Limitations

The compact state representation is useful for a small demonstration, but it has trade-offs:

- It only describes local danger and relative food direction.
- It does not provide the network with the complete board.
- Long-term planning can therefore be difficult.
- Training results can vary between runs because gameplay and food placement are randomized.
- The current project focuses on training rather than a separate inference/evaluation application.

## 🛠️ Possible next steps

Good directions for future development include:

- Add a dedicated evaluation mode.
- Add deterministic seeds and repeatable benchmark runs.
- Move hyperparameters into a configuration file.
- Add automated tests for game mechanics and reward calculation.
- Add periodic checkpoints and resumable training.
- Experiment with Double DQN or Dueling DQN.
- Experiment with prioritized experience replay.
- Compare compact state input against grid/CNN-based state representations.
- Add structured experiment metrics and run summaries.
- Add CI checks for formatting, imports, and tests.

## 🤝 Contributing

Contributions are welcome. Start with [CONTRIBUTING.md](CONTRIBUTING.md), then open an issue or pull request describing the change.

## 📜 License

This project is licensed under the [MIT License](LICENSE).

## 🙏 Acknowledgments

This project is an educational implementation of Deep Q-Learning applied to the classic Snake environment. It is intended as a practical way to explore reinforcement learning, neural networks, reward design, and experience replay.

---

**Repository:** https://github.com/vallabhatech/SnakeAI
