"""
Reinforcement Learning Agent for Snake Game

This module implements a Deep Q-Learning agent that learns to play Snake through
experience replay and neural network-based decision making. The agent observes
the game state, selects actions, and updates its policy based on rewards and
 punishments received during gameplay.
"""

import torch
import random
import numpy as np
from collections import deque
from snake_game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Agent:
    """Deep Q-Learning agent for playing Snake.
    
    This agent implements the core reinforcement learning logic including
    state representation, action selection, experience replay, and training.
    It uses an epsilon-greedy exploration strategy and learns from both
    immediate experiences and replayed memories.
    
    Attributes:
        n_games (int): Number of games played (used for epsilon decay)
        epsilon (float): Exploration rate (randomness in action selection)
        gamma (float): Discount factor for future rewards (0.9)
        memory (deque): Experience replay buffer storing past transitions
        model (Linear_QNet): Neural network for Q-value approximation
        trainer (QTrainer): Handles Q-learning updates and optimization
    """

    def __init__(self):
        """Initialize the agent with default hyperparameters.
        
        Sets up the neural network, experience replay memory, and training
        components. The model is automatically moved to GPU if available.
        """
        self.n_games = 0
        self.epsilon  = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(11, 256, 3).to(device)  # Move model to GPU
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, game):
        """Extract the current game state as an 11-feature binary vector.
        
        Creates a compact state representation that captures essential information
        for decision making: immediate dangers, current direction, and food location.
        This efficient representation allows the neural network to learn quickly
        without processing the entire game board.
        
        Args:
            game (SnakeGameAI): The current game environment instance
        
        Returns:
            np.ndarray: Binary array of 11 features:
                [0-2]: Danger straight, right, left (True if collision would occur)
                [3-6]: Current direction (left, right, up, down)
                [7-10]: Food location relative to head (left, right, up, down)
        
        Note:
            Danger detection considers the current direction to determine which
            adjacent positions correspond to "straight", "right", and "left".
        """
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        """Store a transition in the experience replay memory.
        
        Saves the complete transition (state, action, reward, next_state, done)
        for later training through experience replay. This breaks temporal
        correlations and improves learning stability.
        
        Args:
            state (np.ndarray): Current state representation
            action (list): Action taken in the current state
            reward (float): Reward received after taking the action
            next_state (np.ndarray): State after taking the action
            done (bool): Whether the episode ended after this transition
        
        Note:
            Memory automatically removes oldest transitions when capacity
            (MAX_MEMORY = 100,000) is exceeded using deque's maxlen.
        """
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached


    def train_long_memory(self):
        """Train the model on a batch of experiences from replay memory.
        
        Samples a batch of transitions from memory and performs a training step.
        This allows the agent to learn from past experiences multiple times,
        improving sample efficiency and breaking temporal correlations.
        
        Uses random sampling if memory exceeds BATCH_SIZE, otherwise uses
        all available experiences. This is typically called after each
        game episode to consolidate learning from that episode.
        
        Note:
            The commented-out code shows an alternative approach of training
            on each experience individually, but batch training is used for
            better computational efficiency.
        """
        # sample a batch of experiences from memory if memory is too large
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        """Train the model on a single immediate experience.
        
        Performs a training step on the most recent transition before storing
        it in replay memory. This provides immediate learning feedback while
        the experience is still fresh, complementing the batch training from
        replay memory.
        
        Args:
            state (np.ndarray): Current state representation
            action (list): Action taken in the current state
            reward (float): Reward received after taking the action
            next_state (np.ndarray): State after taking the action
            done (bool): Whether the episode ended after this transition
        
        Note:
            This is called after each action during gameplay for immediate
            learning, while train_long_memory is called after each episode.
        """
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        """Select an action using epsilon-greedy policy.
        
        Balances exploration (random actions) and exploitation (optimal actions)
        based on the current epsilon value. Epsilon decreases linearly as more
        games are played, gradually shifting from exploration to exploitation.
        
        Args:
            state (np.ndarray): Current state representation (11 binary features)
        
        Returns:
            list: One-hot encoded action [1,0,0] for straight, [0,1,0] for right,
                  [0,0,1] for left
        
        Note:
            Epsilon formula: epsilon = 80 - n_games
            - Early training: High epsilon (e.g., 80) → 40% random moves
            - Later training: Low epsilon (e.g., 0) → purely greedy actions
            - Random move probability: epsilon/200
        """
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        
        # random move
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            # greedy move
            state0 = torch.tensor(state, dtype=torch.float).to(device)  # Move tensor to GPU
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    """Main training loop for the Snake AI agent.
    
    Continuously trains the agent by playing games, collecting experiences,
    and updating the neural network. Implements the complete reinforcement
    learning cycle: action selection, environment interaction, immediate
    learning, experience storage, and batch training.
    
    The loop continues indefinitely until the user closes the Pygame window.
    Training progress is visualized in real-time with score tracking.
    
    Training process per game:
    1. Get current state from game environment
    2. Select action using epsilon-greedy policy
    3. Execute action and receive reward, next state, and game status
    4. Train on immediate experience (short memory)
    5. Store experience in replay memory
    6. If game over: train on replayed experiences, update statistics
    
    Note:
        - Automatically saves model when new high score is achieved
        - Tracks both individual scores and running mean scores
        - Updates live matplotlib plot with training progress
    """
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        # if game over
        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()