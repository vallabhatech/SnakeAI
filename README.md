# Snake AI

A reinforcement learning project that trains a deep Q-learning agent to play the classic Snake game using PyTorch. The agent learns to navigate the game board, collect food, and avoid collisions through experience replay and neural network-based decision making.

## Features

- **Deep Q-Learning Implementation**: Uses a neural network to learn optimal policies through Q-learning
- **Experience Replay**: Stores and samples past experiences to improve training stability
- **GPU Acceleration**: Automatically utilizes CUDA when available for faster training
- **Real-time Visualization**: Live training progress with score tracking via Matplotlib
- **Compact State Representation**: Efficient 11-feature state encoding for fast learning
- **Self-Play Training**: Agent continuously improves through autonomous gameplay
- **Model Persistence**: Automatically saves best-performing models

## Demo

[![Watch the Snake AI demo](thumbnail.png)](https://youtu.be/cHPOxD0hUVE)

Click the image to watch the trained agent play.

## How It Works

### Core Architecture

The Snake AI system consists of four main components that work together:

1. **Game Environment** (`snake_game.py`): Pygame-based Snake implementation
2. **Agent** (`agent.py`): Decision-making and learning logic
3. **Neural Network** (`model.py`): Q-network for value function approximation
4. **Visualization** (`helper.py`): Real-time training progress tracking

### AI Decision-Making Process

#### 1. State Representation

The agent perceives the game state through 11 binary features that capture essential information without processing the entire game board:

**Danger Detection (3 features):**
- Danger immediately straight ahead
- Danger immediately to the right  
- Danger immediately to the left

**Current Direction (4 features):**
- Moving left
- Moving right
- Moving up
- Moving down

**Food Location (4 features):**
- Food is to the left of the head
- Food is to the right of the head
- Food is above the head
- Food is below the head

This compact representation allows the agent to make decisions based on immediate threats and goal direction while keeping the neural network input small and efficient.

#### 2. Action Space

The agent chooses from three relative movements rather than absolute directions:

```text
[1, 0, 0] → Continue straight
[0, 1, 0] → Turn right (relative to current direction)
[0, 0, 1] → Turn left (relative to current direction)
```

**Why relative actions?** This design prevents the agent from making illegal 180-degree turns that would cause immediate self-collision. It also allows the same learned policy to work regardless of the snake's absolute orientation, improving generalization.

#### 3. Neural Network Architecture

The Q-network is a simple feedforward neural network:

```
Input Layer (11 neurons) → Hidden Layer (256 neurons, ReLU) → Output Layer (3 neurons)
```

- **Input**: 11 binary state features
- **Hidden Layer**: 256 neurons with ReLU activation for non-linear transformation
- **Output**: 3 Q-values representing the expected future reward for each action

The network outputs Q-values that estimate the long-term reward potential of taking each action from the current state.

#### 4. Q-Learning Algorithm

The agent uses Deep Q-Learning with the following key components:

**Bellman Equation for Q-Value Updates:**
```
Q(s,a) = r + γ * max(Q(s',a'))
```
Where:
- `Q(s,a)` is the Q-value for state-action pair
- `r` is the immediate reward
- `γ` (gamma) is the discount factor (0.9) 
- `max(Q(s',a'))` is the maximum Q-value for the next state

**Experience Replay:**
- Stores transitions (state, action, reward, next_state, done) in a memory buffer
- Capacity: 100,000 transitions
- Randomly samples batches of 1,000 transitions for training
- Breaks correlation between consecutive experiences for more stable learning

**Training Process:**
1. **Short-term Memory**: Updates the network after each action using the immediate experience
2. **Long-term Memory**: After each game episode, trains on a batch of sampled experiences from memory
3. **Loss Function**: Mean Squared Error between predicted Q-values and target Q-values from Bellman equation

#### 5. Exploration vs. Exploitation

The agent uses an epsilon-greedy strategy to balance exploration and exploitation:

```
epsilon = 80 - number_of_games_played
```

- **Exploration**: With probability `epsilon/200`, take a random action
- **Exploitation**: Otherwise, take the action with the highest Q-value

As the agent plays more games, epsilon decreases, reducing random exploration and increasing reliance on learned policies.

#### 6. Reward Function

The agent receives feedback through a sparse reward system:

| Event | Reward | Rationale |
|-------|--------|-----------|
| Eat food | `+10` | Positive reinforcement for achieving the goal |
| Hit wall or body | `-10` | Negative reinforcement for game-ending behavior |
| Normal movement | `0` | No immediate feedback, encourages efficiency |

Additionally, games terminate if the snake goes too long without eating (100 * snake_length steps), preventing infinite loops and encouraging active food-seeking behavior.

### Training Loop

The complete training cycle for each game:

1. **Initialize**: Reset game environment, get initial state
2. **Act**: Agent selects action based on current state (exploration or exploitation)
3. **Execute**: Game performs action, returns reward, game status, and score
4. **Observe**: Agent captures new state after action
5. **Learn (Short-term)**: Update network with immediate experience
6. **Remember**: Store transition in experience replay memory
7. **Repeat**: Continue until game over
8. **Learn (Long-term)**: Train network on batch of replayed experiences
9. **Evaluate**: Update training statistics, save model if record broken
10. **Visualize**: Update live score tracking graph

## Project Structure

```
SnakeAI/
├── agent.py              # Agent class with state encoding, action selection, and training logic
├── model.py              # Neural network architecture and Q-learning trainer implementation
├── snake_game.py         # Pygame environment with game mechanics and collision detection
├── helper.py             # Matplotlib visualization for training progress tracking
├── model/
│   └── model.pth         # Saved neural network weights from best training session
├── requirements.txt      # Python package dependencies
├── README.md             # This file - comprehensive project documentation
├── notes.txt             # Development notes and architecture overview
├── thumbnail.png         # Demo video thumbnail
├── arial.ttf             # Font file for game score display
└── .vscode/
    └── settings.json     # VSCode configuration settings
```

### File Descriptions

- **agent.py**: Core reinforcement learning implementation including the Agent class that manages the training loop, experience replay, and action selection using epsilon-greedy policy
- **model.py**: Contains Linear_QNet (the neural network) and QTrainer (handles Q-learning updates with Bellman equation implementation)
- **snake_game.py**: Implements the SnakeGameAI class providing the game environment, state transitions, collision detection, and reward system
- **helper.py**: Provides live plotting functionality to visualize training progress with individual and mean scores
- **requirements.txt**: Lists all Python dependencies needed to run the project

## Installation and Usage

### Prerequisites

- **Python 3.7 or higher**: The project uses modern Python features and libraries
- **GPU with CUDA support (optional)**: For accelerated training, though CPU fallback is provided

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Jamez43/SnakeAI.git
   cd SnakeAI
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment:**
   
   **On Linux/Mac:**
   ```bash
   source .venv/bin/activate
   ```
   
   **On Windows:**
   ```powershell
   .venv\Scripts\activate
   ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   This will install:
   - `torch`: PyTorch for neural network implementation
   - `torchvision`: Additional PyTorch utilities
   - `numpy`: Numerical computing for state representation
   - `matplotlib`: Plotting library for training visualization
   - `ipython`: Interactive Python utilities for live plotting
   - `pygame`: Game engine for the Snake environment

### Running the Training

Start the training process by running:

```bash
python agent.py
```

This will:
- Launch the Pygame window showing the agent playing Snake
- Open a Matplotlib window displaying training progress (scores and mean scores)
- Continuously train the agent until you close the Pygame window
- Automatically save the best model to `model/model.pth` when a new record is achieved

### Training Tips

- **Initial Performance**: The agent will play randomly at first (high exploration)
- **Learning Curve**: Typically shows improvement after 50-100 games
- **Convergence**: Performance stabilizes as epsilon decreases (after ~80 games)
- **GPU Usage**: If CUDA is available, training will be significantly faster
- **Model Saving**: Best models are automatically saved when records are broken

### Using a Pre-trained Model

To use the included pre-trained model (`model/model.pth`), you would need to modify the code to load the weights without training. The current implementation is designed for training from scratch, but the saved weights can be loaded for inference:

```python
agent = Agent()
agent.model.load_state_dict(torch.load('model/model.pth'))
agent.model.eval()  # Set to evaluation mode
```

## Results and Performance

### Observed Performance

- **Best Score**: Approximately 70 food items in a single game
- **Training Time**: Several hours of continuous training for convergence
- **Learning Rate**: Noticeable improvement after 50-100 games

### Limitations

1. **State Representation**: The 11-feature state only captures immediate danger and relative food direction. It doesn't provide complete board visibility, limiting long-term planning capabilities.

2. **Evaluation**: Current results are observational rather than statistically validated. No fixed-seed evaluation harness or aggregate metrics across multiple runs are included.

3. **Local Optima**: The compact state space can lead to locally optimal behaviors that don't generalize to all game situations.

4. **Memory**: While experience replay helps, the simple network architecture may not capture complex strategies that require remembering longer sequences.

### Possible Improvements

- **Evaluation Mode**: Add a separate evaluation script to test trained models without training
- **Statistical Validation**: Implement fixed-seed evaluation with mean, median, and variance metrics
- **Enhanced State**: Experiment with full-grid convolutional neural networks for complete board awareness
- **Advanced Algorithms**: Implement Double Q-Learning, Dueling networks, or Prioritized Experience Replay
- **Hyperparameter Tuning**: Systematic exploration of learning rates, batch sizes, and network architectures
- **Testing Suite**: Add unit tests for movement, collision detection, and reward calculation
- **Checkpoints**: Save periodic training checkpoints for resumption and analysis
- **Configuration Management**: Externalize hyperparameters to configuration files

## Technical Details

### Hyperparameters

- **Learning Rate**: 0.001 (Adam optimizer)
- **Discount Factor (γ)**: 0.9
- **Batch Size**: 1,000 transitions
- **Memory Capacity**: 100,000 transitions
- **Epsilon Decay**: Linear from 80 to 0 over games
- **Game Speed**: 100 FPS
- **Grid Size**: 640×480 pixels with 20-pixel cells (32×24 grid)

### Dependencies

The project requires the following Python packages (see `requirements.txt`):

```
torch          # Deep learning framework
torchvision    # PyTorch vision utilities
numpy          # Numerical computing
matplotlib     # Plotting and visualization
ipython        # Interactive Python utilities
pygame         # Game development library
```

## License

This project is provided as-is for educational purposes. Please refer to the original repository for specific licensing information.

## Contributing

Contributions are welcome! Areas for improvement include:
- Enhanced evaluation metrics
- Additional network architectures
- Better state representations
- Improved visualization
- Testing infrastructure

## Acknowledgments

This project demonstrates the practical application of Deep Q-Learning to a classic game environment, showcasing how reinforcement learning can be applied to sequential decision-making problems.
