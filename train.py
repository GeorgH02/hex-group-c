# running this file trains a REINFORCE model 
# for a specific board size and saves it as "model_{board_size}.pth"
import torch
import torch.nn as nn
import torch.optim as optim
from hex_engine import hexPosition
from submission.facade_groupC import HexPolicy, board_to_tensor

def train(episodes=5000):
    board_size = int(input("Enter board size: "))
    net = HexPolicy(board_size)
    optimizer = optim.Adam(net.parameters(), lr=1e-3)

    win_count = 0

    for episode in range(episodes):
        game = hexPosition(size=board_size)

        log_probs = []

        while game.winner == 0:
            actions = game.get_action_space()
            
            if game.player == 1:
                state = board_to_tensor(game.board)
                probs = net(state).squeeze(0)

                legal_indices = [row * board_size + col for (row, col) in actions]
                legal_probs = torch.stack([probs[i] for i in legal_indices])
                legal_probs = torch.clamp(legal_probs, min=1e-8)
                legal_probs = legal_probs / legal_probs.sum()

                dist = torch.distributions.Categorical(legal_probs)
                sampled = dist.sample()
                log_probs.append(dist.log_prob(sampled))

                action_idx = legal_indices[sampled.item()]
                move = (action_idx // board_size, action_idx % board_size)
            
            # random opponent
            else:  
                import random
                move = random.choice(actions)

            game.move(move)

        reward = 1.0 if game.winner == 1 else -1.0
        if game.winner == 1:
            win_count += 1

        if log_probs:
            loss = -sum(lp * reward for lp in log_probs)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()

        if episode % 500 == 0:
            winrate = win_count / (episode + 1) * 100
            print(f"Episode {episode}/{episodes}, Win rate: {winrate:.1f}%")
            
    # saving trained model
    torch.save(net.state_dict(), f"submission/model_{board_size}.pth")
    print(f"Model of size {board_size} has been trained and saved")

if __name__ == "__main__":
    train()