"""
Neural Network Model and Q-Learning Trainer

This module implements the neural network architecture for Deep Q-Learning
and the training logic that updates the network using the Bellman equation.
The model is a simple feedforward network that approximates the Q-value function
for the Snake game environment.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Linear_QNet(nn.Module):
    """Feedforward neural network for Q-value approximation in Deep Q-Learning.
    
    A simple fully-connected network that maps game states to Q-values for
    each possible action. The network learns to predict the expected future
    reward for taking each action from a given state.
    
    Architecture:
        Input layer (input_size neurons) → Hidden layer (hidden_size neurons, ReLU) → 
        Output layer (output_size neurons)
    
    For Snake AI: 11 input features → 256 hidden neurons → 3 output Q-values
    
    Attributes:
        linear1 (nn.Linear): First layer mapping input to hidden representation
        linear2 (nn.Linear): Output layer mapping hidden to Q-values
    """

    def __init__(self, input_size, hidden_size, output_size):
        """Initialize the Q-network architecture.
        
        Args:
            input_size (int): Number of input features (11 for Snake AI)
            hidden_size (int): Number of neurons in hidden layer (256)
            output_size (int): Number of output Q-values (3 for straight, right, left)
        """
        super().__init__()
        # Define the layers
        # simple feedforward neural network with one hidden layer
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """Forward pass through the network.
        
        Processes the input state through the network layers to produce
        Q-value predictions for each possible action.
        
        Args:
            x (torch.Tensor): Input state tensor of shape (batch_size, input_size)
        
        Returns:
            torch.Tensor: Q-value predictions of shape (batch_size, output_size)
        
        Note:
            Uses ReLU activation on the hidden layer for non-linear transformation,
            but no activation on the output layer since Q-values can be any real number.
        """
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        """Save the model's state dictionary to a file.
        
        Saves the learned weights and biases to disk for later loading or
        analysis. Creates the model directory if it doesn't exist.
        
        Args:
            file_name (str): Name of the file to save the model to
                           (default: 'model.pth')
        
        Note:
            Only saves the model parameters (state_dict), not the entire model
            object. The model architecture must be recreated when loading.
        """
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    """Trainer class for implementing Q-learning updates on the neural network.
    
    Handles the core Q-learning algorithm including Bellman equation updates,
    loss calculation, and network optimization. Supports both single-step
    updates and batch training for experience replay.
    
    Attributes:
        lr (float): Learning rate for the optimizer (0.001)
        gamma (float): Discount factor for future rewards (0.9)
        model (Linear_QNet): The neural network being trained
        optimizer (optim.Adam): Adam optimizer for weight updates
        loss_function (nn.MSELoss): Mean squared error loss function
    """

    def __init__(self, model, lr, gamma):
        """Initialize the Q-trainer with model and hyperparameters.
        
        Args:
            model (Linear_QNet): The neural network to train
            lr (float): Learning rate for the optimizer
            gamma (float): Discount factor for future rewards (0-1)
        """
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.loss_function = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        """Perform one training step using Q-learning update rule.
        
        Implements the core Q-learning algorithm: updates Q-values based on
        the Bellman equation, computes loss between predicted and target Q-values,
        and backpropagates to update network weights.
        
        Args:
            state (torch.Tensor or np.ndarray): Current state(s)
            action (torch.Tensor or np.ndarray): Action(s) taken
            reward (torch.Tensor or np.ndarray): Reward(s) received
            next_state (torch.Tensor or np.ndarray): Next state(s) after action
            done (bool or tuple): Whether episode(s) terminated
        
        Note:
            Handles both single samples and batches automatically by adding
            batch dimension when needed. Uses the Bellman equation:
            Q(s,a) = r + γ * max(Q(s',a')) for non-terminal states
            Q(s,a) = r for terminal states
        """
        # turn the data into tensors
        state = torch.tensor(state, dtype=torch.float).to(device)
        next_state = torch.tensor(next_state, dtype=torch.float).to(device)
        action = torch.tensor(action, dtype=torch.long).to(device)
        reward = torch.tensor(reward, dtype=torch.float).to(device)

        # check if state is a single sample(not a batch)
        # (x, )
        if len(state.shape) == 1:
            # change dimension to (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # current predictions -> Q(s, a)
        preds = self.model(state)

        # clone predictions to keep original values for loss calculation
        predsClone = preds.clone()
        
        # for each index in the batch run Bellman equation to update Q values
        # V(s) = reward_s + gamma * max Q(next_state, a)
        # Q(s, a) = Q(s, a) + gamma * max Q(s+1, a)
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            # update the Q value for the action taken
            predsClone[idx][torch.argmax(action[idx]).item()] = Q_new

        
        self.optimizer.zero_grad()
        loss = self.loss_function(predsClone, preds)
        loss.backward()
        self.optimizer.step()