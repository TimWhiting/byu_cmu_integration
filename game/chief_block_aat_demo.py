from matplotlib import pyplot as plt
from simple_rl.agents import QLearningAgent
from old_chief_agent_block import PlayerPool, ChiefAgent
from typing import List
import pickle
import numpy as np
from chief_block_aat import create_agents
from block_game_agent import FixedPolicyBlockGameAgent
from block_game import BlockGameMDP, BlockGameState
from pygame.locals import *
import pygame
from typing import Tuple


block_game = BlockGameMDP()
pygame.init()

screen = pygame.display.set_mode([1400, 1280])
pygame.display.set_caption("Block Game")
font = pygame.font.Font(None, 20)

# Game images
red_square_img = pygame.transform.scale(
    pygame.image.load("game/images/red-square.png"), (50, 50))
red_square_rect = red_square_img.get_rect()
red_square_rect.topleft = 425, 230

blue_square_img = pygame.transform.scale(
    pygame.image.load("game/images/blue-square.png"), (50, 50))
blue_square_rect = blue_square_img.get_rect()
blue_square_rect.topleft = 485, 230

purple_square_img = pygame.transform.scale(
    pygame.image.load("game/images/purple-square.png"), (50, 50))
purple_square_rect = purple_square_img.get_rect()
purple_square_rect.topleft = 545, 230

red_triangle_img = pygame.transform.scale(
    pygame.image.load("game/images/red-triangle.png"), (50, 50))
red_triangle_rect = red_triangle_img.get_rect()
red_triangle_rect.topleft = 605, 230

blue_triangle_img = pygame.transform.scale(
    pygame.image.load("game/images/blue-triangle.png"), (50, 50))
blue_triangle_rect = blue_triangle_img.get_rect()
blue_triangle_rect.topleft = 665, 230

purple_triangle_img = pygame.transform.scale(
    pygame.image.load("game/images/purple-triangle.png"), (50, 50))
purple_triangle_rect = purple_triangle_img.get_rect()
purple_triangle_rect.topleft = 725, 230

red_circle_img = pygame.transform.scale(
    pygame.image.load("game/images/red-circle.png"), (50, 50))
red_circle_rect = red_circle_img.get_rect()
red_circle_rect.topleft = 785, 230

blue_circle_img = pygame.transform.scale(
    pygame.image.load("game/images/blue-circle.png"), (50, 50))
blue_circle_rect = blue_circle_img.get_rect()
blue_circle_rect.topleft = 845, 230

purple_circle_img = pygame.transform.scale(
    pygame.image.load("game/images/purple-circle.png"), (50, 50))
purple_circle_rect = purple_circle_img.get_rect()
purple_circle_rect.topleft = 905, 230

# Game buttons for each action and image
buttons = {
    "red square": (red_square_img, red_square_rect), "blue square": (blue_square_img, blue_square_rect),
    "purple square": (purple_square_img, purple_square_rect), "red triangle": (red_triangle_img, red_triangle_rect),
    "blue triangle": (blue_triangle_img, blue_triangle_rect), "purple triangle": (purple_triangle_img, purple_triangle_rect),
    "red circle": (red_circle_img, red_circle_rect), "blue circle": (blue_circle_img, blue_circle_rect),
    "purple circle": (purple_circle_img, purple_circle_rect)
}


def inside_rectangle(rec, point):
    x, y = point

    return rec.left <= x <= (rec.left + rec.width) and rec.top <= y <= (rec.top + rec.height)


def action_choice(action_coords):
    x, y = action_coords

    for button in buttons:
        _, rect = buttons[button]

        if (inside_rectangle(rect, (x, y))):
            return button

    return None


data_dir = './training_data/'


def distance_func(x, y):
    play_num_dist = abs(x[0] - y[0])
    round_num_dist = 2 * abs(x[1] - y[1])
    curr_avg_payoff_dist = 2 * abs(x[2] - y[2])
    human_static_during_round_dist = 2 * abs(x[3] - y[3])
    human_static_across_rounds_dist = 2 * abs(x[4] - y[4])
    human_captured_in_experts_dist = 5 * abs(x[5] - y[5])

    return sum([play_num_dist, round_num_dist, curr_avg_payoff_dist, human_static_during_round_dist, human_static_across_rounds_dist, human_captured_in_experts_dist])


use_new_chief = False

trained_knn_file = 'trained_knn_aat_new_chief.pickle' if use_new_chief else 'trained_knn_aat.pickle'
trained_knn_scaler_file = 'trained_knn_scaler_aat_new_chief.pickle' if use_new_chief else 'trained_knn_scaler_aat.pickle'

model = pickle.load(open(data_dir + trained_knn_file, 'rb'))
scaler = pickle.load(open(data_dir + trained_knn_scaler_file, 'rb'))

training_data_file = 'training_data_new_chief.pickle' if use_new_chief else 'training_data.pickle'

with open(data_dir + training_data_file, 'rb') as f:
    training_data = np.array(pickle.load(f))


def knn_aat_prediction_func(x: List) -> Tuple[List, List, List]:
    x = np.array(x).reshape(1, -1)
    x_scaled = scaler.transform(x)
    neighbor_distances, neighbor_indices = model.kneighbors(x_scaled, 15)

    predictions = []
    corrections = []
    distances = []

    for i in range(len(neighbor_indices[0])):
        neighbor_idx = neighbor_indices[0][i]
        neighbor_dist = neighbor_distances[0][i]
        predictions.append(training_data[neighbor_idx, -1])
        corrections.append(training_data[neighbor_idx, -2])
        distances.append(neighbor_dist)

    return predictions, corrections, distances


chief_idx = 1
human_idx = 0

# Pool for the chief agent
pool_agents = create_agents()
player_pool = PlayerPool(list(pool_agents.values()), sample_size=10)

# Create the chief agent
chief_player = ChiefAgent(actions=block_game.get_actions(
), name='Chief', player_pool=player_pool, partner_idx=human_idx, likelihood_threshold=0.6)

human_name = 'Human'

agents = [human_name, chief_player]

reward_map = {human_name: 0, chief_player.name: 0}

block_game.reset()
state = block_game.get_init_state()

n_rounds = 50

final_predictions = []

for round_num in range(1, n_rounds + 1):
    block_game.reset()
    state = block_game.get_init_state()
    action_map = dict()
    round_predictions = []

    while not state.is_terminal():
        screen.fill([255, 255, 255])

        text_round = font.render(
            "Round {}/{}".format(round_num, n_rounds), 1, (5, 5, 5))
        text_round_rect = text_round.get_rect(centerx=675, centery=10)
        screen.blit(text_round, text_round_rect)

        text1 = font.render("Your selections:", 1, (5, 5, 5))
        text2 = font.render("Opponent's selections:", 1, (5, 5, 5))
        text1rect = text1.get_rect(centerx=950, centery=50)
        text2rect = text2.get_rect(centerx=400, centery=50)

        screen.blit(text1, text1rect)
        screen.blit(text2, text2rect)

        totalrewardtext1 = font.render(
            "Your total reward: " + str(reward_map["Human"]), 1, (5, 5, 5))
        totalrewardtext2 = font.render(
            "Their total reward: " + str(reward_map[chief_player.name]), 1, (5, 5, 5))

        recent_pred = round(
            final_predictions[-1], 2) / 100 if len(final_predictions) > 0 else 'N/A'
        predictedrewardtext = font.render(
            "Chief\'s predicted total reward: " + str(recent_pred), 1, (5, 5, 5))

        text1rect = text1.get_rect(centerx=800, centery=180)
        text2rect = text2.get_rect(centerx=550, centery=180)
        predrect = predictedrewardtext.get_rect(centerx=675, centery=200)

        screen.blit(totalrewardtext1, text1rect)
        screen.blit(totalrewardtext2, text2rect)
        screen.blit(predictedrewardtext, predrect)

        for b in buttons:
            if state.blocks[block_game.actions.index(b)] == BlockGameState.AVAILABLE:
                img, rect = buttons[b]
                pygame.draw.rect(screen, (200, 50, 200), rect)
                screen.blit(img, rect)

        human_coordinate = 850
        chief_coordinate = 300

        for i, player_idx in enumerate(state.blocks):
            # [(img, rect), (img2, rect2)]
            if player_idx == human_idx:
                img, rect = list(buttons.values())[i]
                new_rect = pygame.Rect(human_coordinate, 70, 50, 50)
                screen.blit(img, new_rect)
                human_coordinate += 60

            elif player_idx == chief_idx:
                img, rect = list(buttons.values())[i]
                new_rect = pygame.Rect(chief_coordinate, 70, 50, 50)
                screen.blit(img, new_rect)
                chief_coordinate += 60

        pygame.display.flip()

        agent_action = None

        for agent in agents:
            if agent == "Human":
                # get action
                chosen_action = None
                clicked = False

                while chosen_action is None:
                    for event in pygame.event.get():
                        if event.type == MOUSEBUTTONDOWN:
                            clicked = True
                        if event.type == MOUSEBUTTONUP and clicked:
                            mouse_pos = pygame.mouse.get_pos()
                            chosen_action = action_choice(mouse_pos)
                            # print(chosen_action, mouse_pos)
                            clicked = False

                        if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                            # Quit.
                            pygame.display.quit()

                action_map[agent] = chosen_action

            else:
                agent_reward = reward_map[agent.name]
                agent_action = agent.act(state, agent_reward, round_num - 1)
                action_map[agent.name] = agent_action

                chiefs_proposed_model = agent.get_proposed_model()
                proposed_avg_payoff = chiefs_proposed_model.baseline_payoff if not isinstance(
                    chiefs_proposed_model, QLearningAgent) else -6.0874999999999995
                n_remaining_rounds = n_rounds - round_num
                proposed_payoff_to_go = proposed_avg_payoff * n_remaining_rounds
                proposed_total_payoff = agent_reward + proposed_payoff_to_go

                human_captured_in_experts = max(
                    agent.current_bayesian_values) > 0.5
                human_static_during_round = True if isinstance(
                    chiefs_proposed_model, FixedPolicyBlockGameAgent) else not chiefs_proposed_model.changes_during_round
                human_static_across_rounds = True if isinstance(
                    chiefs_proposed_model, FixedPolicyBlockGameAgent) else not chiefs_proposed_model.changes_across_rounds

                curr_tup = [state.get_play_num() + 1, round_num + 1, agent_reward / proposed_total_payoff if proposed_total_payoff != 0 else agent_reward / 0.000001, int(
                    human_static_during_round), int(human_static_across_rounds), int(human_captured_in_experts), proposed_total_payoff, proposed_total_payoff]

                predictions, corrections, distances = knn_aat_prediction_func(
                    curr_tup[:-2])

                total_payoff_pred = 0
                inverse_distance_sum = 0

                for dist in distances:
                    inverse_distance_sum += 1 / dist if dist != 0 else 1 / 0.000001

                for i in range(len(predictions)):
                    prediction_i = predictions[i]
                    correction_i = corrections[i]
                    distance_i = distances[i]
                    inverse_distance_i = 1 / distance_i if distance_i != 0 else 1 / 0.000001
                    distance_weight = inverse_distance_i / inverse_distance_sum

                    # total_payoff_pred += ((prediction_i *
                    #                       correction_i) * distance_weight)

                    total_payoff_pred += ((proposed_total_payoff *
                                           correction_i) * distance_weight)

                    # total_payoff_pred += (prediction_i *
                    #                       distance_weight)

                round_predictions.append(total_payoff_pred)

        updated_rewards_map, next_state = block_game.execute_agent_action(
            action_map)

        for agent_name, new_reward in updated_rewards_map.items():
            reward_map[agent_name] += new_reward

        state = next_state

    final_predictions.append(sum(round_predictions) / len(round_predictions))

total_payoff = reward_map[chief_player.name]

errors = []

for pred in final_predictions:
    errors.append(abs(total_payoff - pred))

xvals = list(range(len(errors)))

print('-----------------------------')
print(total_payoff)
print()
print(final_predictions)
print()
print(errors)
print('-----------------------------')

plt.plot(xvals, [error / 100 for error in errors])
# plt.plot(xvals, [2270.0 / 100] * len(xvals), color='red')
plt.title(
    'Chief\'s Predictions During Game')
plt.xlabel('Round #')
plt.ylabel('Total Payoff Prediction')
plt.show()
