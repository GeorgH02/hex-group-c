# train.py

"""
Improved REINFORCE training for the Hex agent.

Main improvements:

1. The policy is trained as RED and BLUE.
2. BLUE is transformed into RED perspective.
3. Legal moves are masked.
4. The model is evaluated regularly against random.
5. Only the best model is saved.
6. A small entropy bonus keeps exploration alive.
7. Immediate win/block rules are also used during evaluation through the facade agent.
"""

import random
import torch
import torch.optim as optim
import torch.nn.functional as F

from hex_engine import hexPosition
from submission.facade_groupC import (
    HexPolicy,
    board_to_tensor,
    recode_blue_board_as_red,
    recode_move,
)


def sample_move_from_policy(net, board, action_set, temperature=1.0):
    """
    Sample one legal move from the policy.

    REINFORCE needs sampling because the model must explore.
    During evaluation we use greedy moves instead.
    """
    size = len(board)

    state = board_to_tensor(board)
    logits = net(state).squeeze(0)

    legal_indices = [row * size + col for row, col in action_set]

    mask = torch.full_like(logits, -1e9)
    mask[legal_indices] = 0.0

    masked_logits = (logits + mask) / temperature
    probs = F.softmax(masked_logits, dim=-1)

    dist = torch.distributions.Categorical(probs)
    sampled_idx = dist.sample()

    log_prob = dist.log_prob(sampled_idx)
    entropy = dist.entropy()

    action_idx = sampled_idx.item()
    move = (action_idx // size, action_idx % size)

    return move, log_prob, entropy


def greedy_move_from_policy(net, board, action_set):
    """
    Pick the highest-scoring legal move.
    Used for evaluation, not training.
    """
    size = len(board)

    with torch.no_grad():
        state = board_to_tensor(board)
        logits = net(state).squeeze(0)

        legal_indices = [row * size + col for row, col in action_set]

        mask = torch.full_like(logits, -1e9)
        mask[legal_indices] = 0.0

        best_idx = torch.argmax(logits + mask).item()

    return best_idx // size, best_idx % size


def get_model_view(game, agent_color):
    """
    Convert current game state into the model's perspective.

    The neural network always learns as RED.
    If the controlled agent is BLUE, we transform the board and actions.
    """
    board_size = game.size
    board = game.board
    actions = game.get_action_space()

    if agent_color == 1:
        return board, actions

    recoded_board = recode_blue_board_as_red(board)
    recoded_actions = [recode_move(action, board_size) for action in actions]

    return recoded_board, recoded_actions


def train_one_episode(net, optimizer, board_size, agent_color):
    """
    Play one episode against a random opponent.

    agent_color:
        1  = train model as RED
        -1 = train model as BLUE
    """
    game = hexPosition(size=board_size)

    log_probs = []
    entropies = []

    while game.winner == 0:
        actions = game.get_action_space()

        if game.player == agent_color:
            model_board, model_actions = get_model_view(game, agent_color)

            model_move, log_prob, entropy = sample_move_from_policy(
                net,
                model_board,
                model_actions,
                temperature=1.0,
            )

            if agent_color == -1:
                move = recode_move(model_move, board_size)
            else:
                move = model_move

            log_probs.append(log_prob)
            entropies.append(entropy)

        else:
            move = random.choice(actions)

        game.move(move)

    reward = 1.0 if game.winner == agent_color else -1.0

    if log_probs:
        policy_loss = -sum(log_probs) * reward

        # Entropy bonus prevents the model from becoming too deterministic too early.
        entropy_bonus = sum(entropies) * 0.01

        loss = policy_loss - entropy_bonus

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()

    return game.winner == agent_color


def evaluate_policy(net, board_size, games_per_side=100):
    """
    Evaluate model against random as RED and BLUE.

    Returns:
        overall win rate, red win rate, blue win rate
    """
    total_wins = 0
    total_games = 0

    red_wins = 0
    blue_wins = 0

    for agent_color in [1, -1]:
        for _ in range(games_per_side):
            game = hexPosition(size=board_size)

            while game.winner == 0:
                actions = game.get_action_space()

                if game.player == agent_color:
                    model_board, model_actions = get_model_view(game, agent_color)
                    model_move = greedy_move_from_policy(net, model_board, model_actions)

                    if agent_color == -1:
                        move = recode_move(model_move, board_size)
                    else:
                        move = model_move
                else:
                    move = random.choice(actions)

                game.move(move)

            total_games += 1

            if game.winner == agent_color:
                total_wins += 1

                if agent_color == 1:
                    red_wins += 1
                else:
                    blue_wins += 1

    overall = total_wins / total_games
    red_rate = red_wins / games_per_side
    blue_rate = blue_wins / games_per_side

    return overall, red_rate, blue_rate


def train(episodes=50000):
    board_size = int(input("Enter board size: "))

    net = HexPolicy(board_size)

    # Lower learning rate gives more stable REINFORCE updates.
    optimizer = optim.Adam(net.parameters(), lr=1e-4)

    best_eval = 0.0

    train_red_wins = 0
    train_blue_wins = 0
    train_red_games = 0
    train_blue_games = 0

    for episode in range(1, episodes + 1):
        # Alternating color training is important.
        # The final agent can be used as RED or BLUE.
        agent_color = 1 if episode % 2 == 0 else -1

        won = train_one_episode(
            net=net,
            optimizer=optimizer,
            board_size=board_size,
            agent_color=agent_color,
        )

        if agent_color == 1:
            train_red_games += 1
            train_red_wins += int(won)
        else:
            train_blue_games += 1
            train_blue_wins += int(won)

        if episode % 1000 == 0:
            train_red_rate = train_red_wins / max(1, train_red_games)
            train_blue_rate = train_blue_wins / max(1, train_blue_games)

            eval_rate, eval_red, eval_blue = evaluate_policy(
                net=net,
                board_size=board_size,
                games_per_side=100,
            )

            print(
                f"Episode {episode}/{episodes} | "
                f"Train RED: {train_red_rate * 100:.1f}% | "
                f"Train BLUE: {train_blue_rate * 100:.1f}% | "
                f"Eval overall: {eval_rate * 100:.1f}% | "
                f"Eval RED: {eval_red * 100:.1f}% | "
                f"Eval BLUE: {eval_blue * 100:.1f}%"
            )

            if eval_rate > best_eval:
                best_eval = eval_rate
                torch.save(net.state_dict(), f"submission/model_{board_size}.pth")
                print(f"New best model saved: {best_eval * 100:.1f}%")

            if best_eval >= 0.90:
                print("Target reached: 90%+ win rate against random.")
                break

    print(f"Training done. Best eval win rate: {best_eval * 100:.1f}%")


if __name__ == "__main__":
    train()