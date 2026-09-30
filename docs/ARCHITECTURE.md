# Architecture

## Overview

Snake AI is organized as a small reinforcement-learning system with four primary runtime components:

```text
SnakeGameAI
    │
    │ state / reward / terminal signal
    ▼
Agent
    │
    ├── get_state()
    ├── get_action()
    ├── remember()
    ├── train_short_memory()
    └── train_long_memory()
    │
    ▼
Linear_QNet + QTrainer
    │
    └── Q-values → selected action
```

## 1. Environment — `snake_game.py`

`SnakeGameAI` provides the environment in which the agent learns.

The environment is responsible for:

- Maintaining the snake body and current direction.
- Placing food on the grid.
- Applying relative actions.
- Detecting wall and self collisions.
- Calculating rewards.
- Ending an episode after a collision or excessive movement without eating.
- Rendering the current game state with Pygame.

### Environment step

`play_step(action)` returns:

```text
(reward, game_over, score)
```

The current reward scheme is:

| Event | Reward |
|---|---:|
| Eat food | +10 |
| Collision / episode termination | -10 |
| Normal movement | 0 |

## 2. State encoder — `agent.py`

The agent does not pass the entire game board to the neural network.

Instead, `get_state()` creates an 11-element binary vector:

```text
[ danger_straight,
  danger_right,
  danger_left,
  moving_left,
  moving_right,
  moving_up,
  moving_down,
  food_left,
  food_right,
  food_up,
  food_down ]
```

This representation is intentionally compact. It makes the model inexpensive to train, while sacrificing complete board awareness.

## 3. Action policy

The agent chooses one of three relative actions:

```text
[1, 0, 0] → straight
[0, 1, 0] → right
[0, 0, 1] → left
```

The game converts these relative actions into absolute directions based on the snake's current orientation.

This avoids explicit 180-degree turns.

## 4. Q-network — `model.py`

The neural network is:

```text
11 inputs
   │
   ▼
256 hidden neurons
   │
  ReLU
   │
   ▼
3 Q-values
```

Each output represents the estimated future value of one available action.

The agent selects the action with the largest predicted Q-value during exploitation.

## 5. Q-learning update

For a non-terminal transition, the implementation uses the Bellman-style target:

```text
Q_target = reward + gamma * max(Q(next_state))
```

For a terminal transition:

```text
Q_target = reward
```

The implementation uses:

- Adam optimizer
- Mean Squared Error loss
- Learning rate = 0.001
- Gamma = 0.9

## 6. Experience replay

Each transition is stored as:

```text
(state, action, reward, next_state, done)
```

The replay buffer can hold up to 100,000 transitions.

After an episode ends, the agent samples up to 1,000 stored transitions and trains on them together.

This reduces the dependence between consecutive experiences and lets the model learn repeatedly from earlier situations.

## 7. Exploration strategy

The agent uses an epsilon-greedy policy.

The current implementation derives epsilon from the number of completed games:

```text
epsilon = 80 - n_games
```

A random action is selected when a random value falls below the current exploration threshold. Otherwise, the network's highest-valued action is selected.

This gradually shifts the agent from exploration toward exploitation.

## 8. Training lifecycle

For each action:

```text
1. Read current state
2. Select action
3. Step environment
4. Read next state
5. Train on short-term transition
6. Store transition in replay memory
```

When the episode ends:

```text
7. Reset environment
8. Increment game counter
9. Train on replay memory
10. Update score statistics
11. Save model if a new record is reached
```

## 9. Persistence

The network weights are saved as:

```text
model/model.pth
```

Only the model state dictionary is persisted. The Python code recreates the network architecture when loading the weights.

## Design trade-offs

### Why use an 11-feature state?

It keeps the project approachable and computationally light. A learner can understand the entire state pipeline without needing a convolutional architecture.

### What does it give up?

The network cannot see the complete board. It therefore has limited information about distant obstacles, body layout, and long-term routes.

A future CNN-based implementation could represent the complete grid and learn richer spatial patterns.

## Future architecture directions

Potential extensions include:

- Separate `train.py` and `evaluate.py` entry points.
- Configurable hyperparameters.
- Checkpoint/resume support.
- Deterministic evaluation runs.
- Double DQN.
- Dueling DQN.
- Prioritized experience replay.
- CNN-based board representation.
- Experiment logging and metric export.
