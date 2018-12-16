import numpy as np
from enum import Enum


class Reward(Enum):
    WIN = 100
    LOSE = -100
    DRAW = 25
    assert(WIN == -LOSE)


ttt_index = np.array([[0, 1, 2],
                      [3, 4, 5],
                      [6, 7, 8]])


def compute_reward(action_index, next_state_desc, player):
    # action_index = 8
    # next_state_desc = np.array(list('--x--x--x'))
    # player = True
    next_state_desc = np.array(list(next_state_desc))
    cur_player = 'x' if player else 'o'
    opponent = 'x' if (cur_player == 'o') else 'o'
    row = ttt_index[int(action_index / 3), :]
    row_elem = next_state_desc[row]
    col = ttt_index[:, action_index % 3]
    col_elem = next_state_desc[col]
    rewards = []

    def reward(action_index, triple_indices):
        triple = next_state_desc[triple_indices]
        triple_action_index = list(triple_indices).index(action_index)
        # print(triple_indices, triple, opponent, action_index)
        if triple[0] == triple[1] == triple[2] == cur_player:
            rewards.append(Reward.WIN.value)
        elif triple[(triple_action_index + 1) % 3] == triple[(triple_action_index + 2) % 3] == opponent:
            rewards.append(Reward.DRAW.value)

    if action_index == 4:
        reward(action_index, np.array([0, 4, 8]))
        reward(action_index, np.array([2, 4, 6]))
    elif action_index % 4 == 0:
        reward(action_index, np.array([0, 4, 8]))
    elif action_index % 2 == 0:
        reward(action_index, np.array([2, 4, 6]))

    reward(action_index, row)
    reward(action_index, col)

    # print(rewards, cur_player, opponent)

    return max(rewards) if rewards else 0


# print(compute_reward(5, '--o--x--o', True))
