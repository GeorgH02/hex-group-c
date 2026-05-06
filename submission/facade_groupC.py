#trivial solution
# def agent (board, action_set):
#     return action_set[0]

#Here should be the necessary Python wrapper for your model, in the form of a callable agent, such as above.
#Please make sure that the agent does actually work with the provided Hex module.

# implementing REINFORCE as baseline agent
import torch
import torch.nn as nn
import numpy as np
import os

# class defining network structure
class HexPolicy(nn.Module):
    def __init__(self, board_size):
        super().__init__()
        self.board_size = board_size
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU())
        self.head = nn.Linear(64 * board_size**2, board_size**2)

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(1)
        return torch.softmax(self.head(x), dim=-1)

def board_to_tensor(board):
    board = np.array(board)
    ch1 = (board == 1).astype(np.float32)    
    ch2 = (board == -1).astype(np.float32)   
    ch3 = (board == 0).astype(np.float32)    
    tensor = np.stack([ch1, ch2, ch3], axis=0)
    return torch.tensor(tensor).unsqueeze(0)  


# loading the model for the chosen board size 
# or playing randomly if model for board size does not exist
MODELS = {}
def get_model(board_size):
    if board_size not in MODELS:
        path = os.path.join(os.path.dirname(__file__), f"model_{board_size}.pth")
        if not os.path.exists(path):
            MODELS[board_size] = None
            print(f"No trained model found for board size {board_size}, agent plays randomly")
        else:
            net = HexPolicy(board_size)
            net.load_state_dict(torch.load(path, map_location="cpu"))
            net.eval()
            MODELS[board_size] = net
    return MODELS[board_size]

def agent(board, action_set):
    import random
    size = len(board)

    state = board_to_tensor(board)
    net = get_model(size)
    
    if net is None:
        return random.choice(action_set)

    with torch.no_grad():
        probs = net(state).squeeze(0)

    legal_indices = [row * size + col for (row, col) in action_set]
    legal_probs = torch.stack([probs[i] for i in legal_indices])
    legal_probs = torch.clamp(legal_probs, min=1e-8)
    legal_probs = legal_probs / legal_probs.sum()

    best = torch.argmax(legal_probs).item()
    action_idx = legal_indices[best]
    return (action_idx // size, action_idx % size)