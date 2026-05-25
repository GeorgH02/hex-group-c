# evaluate.py

from hex_engine import hexPosition
from submission.facade_groupC import agent


def evaluate(n_games=100, board_size=7):
    red_wins = 0
    blue_wins = 0

    for _ in range(n_games):
        game = hexPosition(size=board_size)
        winner = game.machine_vs_machine(
            machine1=agent,
            machine2=None,
        )

        if winner == 1:
            red_wins += 1

    for _ in range(n_games):
        game = hexPosition(size=board_size)
        winner = game.machine_vs_machine(
            machine1=None,
            machine2=agent,
        )

        if winner == -1:
            blue_wins += 1

    red_rate = red_wins / n_games * 100
    blue_rate = blue_wins / n_games * 100
    overall = (red_wins + blue_wins) / (2 * n_games) * 100

    print(f"Agent as RED vs random BLUE:  {red_rate:.1f}%")
    print(f"Agent as BLUE vs random RED: {blue_rate:.1f}%")
    print(f"Overall win rate:            {overall:.1f}%")


if __name__ == "__main__":
    evaluate()