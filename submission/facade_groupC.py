
#trivial solution
# def agent (board, action_set):
#     return action_set[0]

#Here should be the necessary Python wrapper for your model, in the form of a callable agent, such as above.
#Please make sure that the agent does actually work with the provided Hex module.

# implementing REINFORCE as baseline agent

# submission/facade_groupC.py

"""
Group C Hex Agent.

This agent combines:
1. Immediate tactical rules:
   - win if possible
   - block opponent's immediate win if necessary

2. A trained REINFORCE policy network.

3. A shortest-path Hex heuristic:
   - prefer moves that shorten our connection path
   - prefer moves that make the opponent's connection path harder

The policy network always thinks from RED perspective.
RED connects left-to-right.
BLUE connects top-to-bottom.

When the real player is BLUE, the board is transformed so BLUE looks like RED.
The selected move is then transformed back.
"""

import os
import random
import heapq
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn


EMPTY = 0
RED = 1
BLUE = -1

# Turn this off if the heuristic makes performance worse.
USE_HEURISTIC = True

# Try values like 1.0, 1.5, 2.0, 2.5.
# Since your model is already strong, start with 1.0 or 1.5.
HEURISTIC_WEIGHT = 1.5


class HexPolicy(nn.Module):
    """
    Convolutional policy network for Hex.

    Input:
        3 channels:
        channel 0 = own stones
        channel 1 = opponent stones
        channel 2 = empty cells

    Output:
        one score/logit for every board cell.
        Invalid moves are masked outside the network.
    """

    def __init__(self, board_size):
        super().__init__()
        self.board_size = board_size

        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(64 * board_size * board_size, 256),
            nn.ReLU(),
            nn.Linear(256, board_size * board_size),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(1)
        return self.head(x)


def infer_player(board):
    """
    Infer whose turn it is.

    RED starts.
    If both players have the same number of stones, RED moves.
    Otherwise BLUE moves.
    """
    arr = np.array(board)
    red_count = np.sum(arr == RED)
    blue_count = np.sum(arr == BLUE)

    if red_count == blue_count:
        return RED
    return BLUE


def recode_blue_board_as_red(board):
    """
    Transform a BLUE-position into RED perspective.

    The board is flipped along the anti-diagonal and colors are swapped.
    This makes BLUE's top-bottom goal equivalent to RED's left-right goal.
    """
    size = len(board)
    new_board = [[EMPTY for _ in range(size)] for _ in range(size)]

    for row in range(size):
        for col in range(size):
            value = board[size - 1 - col][size - 1 - row]

            if value == RED:
                new_board[row][col] = BLUE
            elif value == BLUE:
                new_board[row][col] = RED
            else:
                new_board[row][col] = EMPTY

    return new_board


def recode_move(move, size):
    """
    Transform coordinates between original and recoded board.
    This operation is its own inverse.
    """
    row, col = move
    return size - 1 - col, size - 1 - row


def board_to_tensor(board):
    """
    Convert board to neural-network input.

    The model always receives the board from the current player's perspective:
        1  = own stones
        -1 = opponent stones
        0  = empty
    """
    board = np.array(board)

    own = (board == RED).astype(np.float32)
    opp = (board == BLUE).astype(np.float32)
    empty = (board == EMPTY).astype(np.float32)

    tensor = np.stack([own, opp, empty], axis=0)
    return torch.tensor(tensor).unsqueeze(0)


def get_neighbors(pos, size):
    """
    Return all valid neighboring hex cells.
    """
    row, col = pos

    candidates = [
        (row - 1, col),
        (row + 1, col),
        (row, col - 1),
        (row, col + 1),
        (row - 1, col + 1),
        (row + 1, col - 1),
    ]

    return [
        (r, c)
        for r, c in candidates
        if 0 <= r < size and 0 <= c < size
    ]


def has_winning_path(board, player):
    """
    Check if a player has connected their two sides.

    RED must connect left to right.
    BLUE must connect top to bottom.
    """
    size = len(board)
    visited = set()
    stack = []

    if player == RED:
        for row in range(size):
            if board[row][0] == RED:
                stack.append((row, 0))
                visited.add((row, 0))

        while stack:
            pos = stack.pop()
            row, col = pos

            if col == size - 1:
                return True

            for nxt in get_neighbors(pos, size):
                if nxt not in visited and board[nxt[0]][nxt[1]] == RED:
                    visited.add(nxt)
                    stack.append(nxt)

    else:
        for col in range(size):
            if board[0][col] == BLUE:
                stack.append((0, col))
                visited.add((0, col))

        while stack:
            pos = stack.pop()
            row, col = pos

            if row == size - 1:
                return True

            for nxt in get_neighbors(pos, size):
                if nxt not in visited and board[nxt[0]][nxt[1]] == BLUE:
                    visited.add(nxt)
                    stack.append(nxt)

    return False


def would_win(board, move, player):
    """
    Check whether placing a stone on move wins immediately.
    """
    temp = deepcopy(board)
    row, col = move
    temp[row][col] = player
    return has_winning_path(temp, player)


def find_immediate_win(board, action_set, player):
    """
    Return a move that wins immediately, if one exists.
    """
    for move in action_set:
        if would_win(board, move, player):
            return move
    return None


def center_preference(move, size):
    """
    Small deterministic fallback preference.

    Random openings are often weak in Hex.
    Central moves are usually more useful than corner moves.
    """
    row, col = move
    center = (size - 1) / 2
    return -((row - center) ** 2 + (col - center) ** 2)


def shortest_path_distance_red_perspective(board, player):
    """
    Estimate how close a player is to connecting left-to-right.

    This function assumes the board is already in RED perspective.

    Cost idea:
        own stone = 0
        empty cell = 1
        opponent stone = blocked

    Lower distance means the player is closer to winning.
    """
    size = len(board)
    INF = 10**9

    dist = [[INF for _ in range(size)] for _ in range(size)]
    pq = []

    # Start from the left side.
    for row in range(size):
        cell = board[row][0]

        if cell == -player:
            continue

        start_cost = 0 if cell == player else 1
        dist[row][0] = start_cost
        heapq.heappush(pq, (start_cost, row, 0))

    while pq:
        current_cost, row, col = heapq.heappop(pq)

        if current_cost != dist[row][col]:
            continue

        # Reached the right side.
        if col == size - 1:
            return current_cost

        for nr, nc in get_neighbors((row, col), size):
            cell = board[nr][nc]

            if cell == -player:
                continue

            step_cost = 0 if cell == player else 1
            new_cost = current_cost + step_cost

            if new_cost < dist[nr][nc]:
                dist[nr][nc] = new_cost
                heapq.heappush(pq, (new_cost, nr, nc))

    return INF


def heuristic_move_score(board, move):
    """
    Score one move from RED perspective.

    Positive score = better for current player.

    In model perspective:
        RED = us
        BLUE = opponent
    """
    size = len(board)
    row, col = move

    temp = deepcopy(board)
    temp[row][col] = RED

    my_distance = shortest_path_distance_red_perspective(temp, RED)
    opponent_distance = shortest_path_distance_red_perspective(temp, BLUE)

    center_bonus = center_preference(move, size) * 0.03

    # We want:
    # - our path distance to become smaller
    # - opponent path distance to become larger
    return (-2.0 * my_distance) + (1.0 * opponent_distance) + center_bonus


MODELS = {}


def get_model(board_size):
    """
    Load the trained model for the given board size.

    If no model exists, the agent still works by using tactical rules
    and a center-based fallback.
    """
    if board_size not in MODELS:
        path = os.path.join(os.path.dirname(__file__), f"model_{board_size}.pth")

        if not os.path.exists(path):
            MODELS[board_size] = None
        else:
            net = HexPolicy(board_size)
            net.load_state_dict(torch.load(path, map_location="cpu"))
            net.eval()
            MODELS[board_size] = net

    return MODELS[board_size]


def choose_policy_move(net, board, action_set):
    """
    Choose a legal move using:
    1. trained REINFORCE policy
    2. optional shortest-path heuristic

    The board is already in RED perspective here.
    """
    size = len(board)

    with torch.no_grad():
        state = board_to_tensor(board)
        logits = net(state).squeeze(0)

        legal_indices = [row * size + col for row, col in action_set]

        mask = torch.full_like(logits, -1e9)
        mask[legal_indices] = 0.0

        masked_logits = logits + mask

        legal_logits = torch.tensor(
            [masked_logits[row * size + col].item() for row, col in action_set],
            dtype=torch.float32
        )

        # Normalize neural-network scores so they combine better with heuristic scores.
        if len(legal_logits) > 1:
            legal_logits = (legal_logits - legal_logits.mean()) / (legal_logits.std() + 1e-6)

        best_move = None
        best_score = -10**18

        for i, move in enumerate(action_set):
            model_score = legal_logits[i].item()

            if USE_HEURISTIC:
                h_score = heuristic_move_score(board, move)
                final_score = model_score + HEURISTIC_WEIGHT * h_score
            else:
                final_score = model_score

            if final_score > best_score:
                best_score = final_score
                best_move = move

    return best_move


def fallback_move(action_set, size):
    """
    Used if no trained model exists.
    Chooses a legal move close to the center.
    """
    return max(action_set, key=lambda move: center_preference(move, size))


def agent(board, action_set):
    """
    Required external function for the Hex engine.

    The engine calls:
        agent(board, action_set) -> move
    """
    size = len(board)
    current_player = infer_player(board)
    opponent = -current_player

    # 1. Win immediately if possible.
    winning_move = find_immediate_win(board, action_set, current_player)
    if winning_move is not None:
        return winning_move

    # 2. Block opponent's immediate win.
    blocking_move = find_immediate_win(board, action_set, opponent)
    if blocking_move is not None:
        return blocking_move

    # 3. Use trained REINFORCE model.
    net = get_model(size)

    if net is None:
        return fallback_move(action_set, size)

    if current_player == RED:
        return choose_policy_move(net, board, action_set)

    # BLUE gets converted into RED perspective.
    recoded_board = recode_blue_board_as_red(board)
    recoded_actions = [recode_move(move, size) for move in action_set]

    recoded_move = choose_policy_move(net, recoded_board, recoded_actions)

    return recode_move(recoded_move, size)