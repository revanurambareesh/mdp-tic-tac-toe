from enum import Enum
from pprint import pprint
from typing import NewType, List, Dict

import numpy as np

from reward_computer import Reward
from reward_computer import compute_reward as cr

import json
import sys

# Define types
State_Desc = NewType('State_Desc', str)
Player = NewType('Player', bool)
Qvalue_Frequency = NewType('Qvalue_Frequency', tuple)
RewardType = NewType('Reward', np.float32)

lam = 0.9
states = {}  # State_Desc: State
state_policy = {}
state_action_expectation = {}  # for debugging

init_state_desc = '---------'
num_episodes = 100000001

ttt_index = np.array([[0, 1, 2],
                      [3, 4, 5],
                      [6, 7, 8]])


class Action(Enum):
    Top_Left = 0
    Top_Center = 1
    Top_Right = 2
    Middle_Left = 3
    Middle_Center = 4
    Middle_Right = 5
    Bottom_Left = 6
    Bottom_Center = 7
    Bottom_Right = 8
    # state_desc actions '012345678'


class State:
    def __init__(self, player: Player, state_desc: State_Desc):
        # print(state_desc)
        assert (len(state_desc) == 9)
        self.possible_actions: List = [action[1].name for action in enumerate(Action) if state_desc[action[0]] == '-']
        self.p: Dict = {action: Qvalue_Frequency((RewardType(0), 0)) for action in self.possible_actions}
        self.v_star = -np.inf
        self.state_desc: State_Desc = state_desc
        self.policy = None  # Represents best action to be taken
        self.player: Player = player

    def __str__(self) -> State_Desc:
        print('\n\n--------\n')
        print(self.p, 'x' if self.player else 'o', 'Policy', self.policy)
        return Util.prettify_game(self.state_desc)

    def update_q(self, action: Action, reward: Reward):
        q, f = self.p[action]
        self.p[action] = (q + reward, f + 1)

    def update_policy(self):
        best_action = None
        best_q_by_f = -1000
        for action in self.p.keys():
            # print(self.p.keys())
            q, f = self.p[action]
            if f == 0: continue
            self.p[action] = (q / f), 1
            if (q / f) > best_q_by_f:
                best_q_by_f = q / f
                best_action = action

        # print(best_action, self.possible_actions, self.state_desc)
        if best_action == -1 and self.possible_actions:
            self.policy = Action[np.random.choice(self.possible_actions)].value
        elif best_action:
            self.v_star = best_q_by_f
            self.policy = Action[best_action].value


class Util:
    @staticmethod
    def add_state_to_state_desc(state: State):
        # add a state to the dict if the state is not already present
        pass

    @staticmethod
    def compute_policy_for_all_states():
        for state in states.values():
            state.update_policy()
            state_policy[state.state_desc] = state.policy

    @staticmethod
    def is_terminal_state(state: State) -> bool:
        return '-' not in state.state_desc

    @staticmethod
    def get_state(player: Player or None, state_desc: State_Desc) -> State:
        if state_desc not in states.keys():
            states[state_desc] = State(player, state_desc)
        return states[state_desc]

    @staticmethod
    def pretty_print_policy():
        for state_desc in state_policy.keys():
            state = Util.get_state(None, state_desc)
            game = Util.prettify_game(state_desc)
            print('-----------------------')
            print(game)
            print('player', 'x' if state.player else 'o', 'policy', state_policy[state_desc])
            print('-----------------------\n\n\n')

    @staticmethod
    def prettify_game(game: State_Desc):
        return game[0:3] + '\n' + game[3:6] + '\n' + game[6:9]


def play_till_end(state: State) -> tuple:
    player = state.player
    action = np.random.choice(state.possible_actions)
    action_index = Action[action].value
    next_state_desc = state.state_desc[:action_index] + ('x' if player else 'o') + state.state_desc[action_index + 1:]

    # print('player', player, 'state_desc', state.state_desc, 'move', action, 'index', action_index)

    reward = cr(action_index, next_state_desc, player)
    if reward == Reward.WIN.value or '-' not in next_state_desc:  # terminal states
        new_player1_reward, new_player2_reward = (1 if player else -1) * reward, (-1 if player else 1) * reward
    else:
        (player1_reward, player2_reward) = play_till_end(Util.get_state(not player, next_state_desc))
        (new_player1_reward, new_player2_reward) = (reward + lam * player1_reward if player else player1_reward,
                                                    player2_reward if player else reward + lam * player2_reward)
    # print('new reward', (new_player1_reward, new_player2_reward))

    state.update_q(action, new_player1_reward if player else new_player2_reward)
    # print('State', state, (new_player1_reward, new_player2_reward))
    return new_player1_reward, new_player2_reward


def update_progress(cur, upper):
    progress = (cur+1)/upper * 100
    hash = '#' * int(progress / 2)
    dash = '-' * (50 - len(hash))
    sys.stdout.write('\r[{0}] ({1} of {2}) {3}%'.format(hash + dash, cur+1, upper, int(progress)))


def learn():
    print('Learning to Play Tic Tac Toe')
    for i in range(num_episodes):
        play_till_end(Util.get_state(True, init_state_desc))
        update_progress(i, num_episodes)

    Util.compute_policy_for_all_states()


def get_next_move(prev_player, state_desc):
    if prev_player:
        if state_desc in state_policy.keys():
            state = Util.get_state(None, state_desc)
            if '--debug' in sys.argv:
                print(state_action_expectation[state.state_desc])
            action_index = state_policy[state_desc]
        else:
            if '--debug' in sys.argv:
                print('No policy found')
            possible_actions = [i[0] for i in enumerate(list(state_desc)) if i[1] == '-']
            action_index = np.random.choice(possible_actions)
    else:
        action_index = int(input('Enter TTT index' + str([i[0] for i in enumerate(list(state_desc)) if i[1] == '-'])))
    return not prev_player, action_index


def play():
    cur_state_desc = init_state_desc
    human = 'x'
    comp = 'x' if human == 'o' else 'o'
    human_first = True
    did_human_play = not human_first

    while True:
        print(Util.prettify_game(cur_state_desc))
        did_human_play, action_index = get_next_move(did_human_play, cur_state_desc)
        next_state_desc = cur_state_desc[:action_index] + (human if did_human_play else comp) + cur_state_desc[
                                                                                                action_index + 1:]
        reward = cr(action_index, next_state_desc, did_human_play)

        if reward == Reward.WIN.value:
            print('\n\n')
            print(Util.prettify_game(next_state_desc))
            print('Game ended', 'You Win' if did_human_play else 'Computer Wins')
            return
        if '-' not in next_state_desc:
            print('\n\n')
            print(Util.prettify_game(next_state_desc))
            print('Game drawn')
            return

        cur_state_desc = next_state_desc

        print('----------\n\n\n')


if '--learn' in sys.argv:
    learn()
    with open(str(num_episodes) + 'policy.json', 'w') as fp:
        json.dump(state_policy, fp)

    print('\n\nSaving information for debugging.')
    for index, state in enumerate(states.values()):
        state_action_expectation[state.state_desc] = state.p
        with open(str(num_episodes) + 'debug.json', 'w') as fp:
            json.dump(state_action_expectation, fp)
        update_progress(index, len(states))

# for i in states.values():
#     print(i)
# Util.pretty_print_policy()

with open(str(num_episodes) + 'policy.json', 'r') as fp:
    state_policy = json.load(fp)

if '--debug' in sys.argv:
    with open(str(num_episodes) + 'debug.json', 'r') as fp:
        state_action_expectation = json.load(fp)

# pprint(state_policy)

while True:
    play()
    print('\n\n\n\n\n\nNewGame\n')
