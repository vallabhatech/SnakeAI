import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # Define the layers
        # simple feedforward neural network with one hidden layer
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    # relu activation function for hidden layer
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    # save the model to a file
    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.loss_function = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
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