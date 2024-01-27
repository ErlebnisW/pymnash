import numpy as np
from pymnash.game import Game
import itertools

def generate_team_payoff(num_players_per_team, num_actions_per_player, seed):
    """
    Generate a random zero-sum payoff tensor for two teams with the same number of players
    and the same number of actions per player.
    """
    np.random.seed(seed)
    # Determine the total number of players and actions
    n = num_players_per_team * 2
    m = num_actions_per_player

    # Initialize the payoff with random values for team1 players (e.g., team 1)
    payoff_shape = [m] * n
    team_1_payoff = np.random.randint(low=0, high=5, size=payoff_shape + [num_players_per_team])

    team_2_payoff = -team_1_payoff

    # Concatenate the two halves to form the complete payoff tensor
    payoff = np.concatenate((team_1_payoff, team_2_payoff), axis=-1)

    return payoff

def calculate_expected_payoffs(num_players, num_actions_per_player, payoff_tensor, strategies):
    expected_payoffs = np.zeros(num_players)

    for action_combination in itertools.product(*[range(num_actions_per_player) for _ in range(num_players)]):
        prob = 1
        for i, action in enumerate(action_combination):
            prob *= strategies[i].get(action, 0)

        # 计算并累加每个玩家的期望收益
        for i in range(num_players):
            expected_payoffs[i] += prob * payoff_tensor[tuple(list(action_combination) + [i])]

    return expected_payoffs


def calculate_pure_cooperative_deviation_payoffs(num_players_per_team, num_actions_per_player, payoff_tensor, equilibrium):
    best_deviation_payoff = np.zeros(num_players_per_team*2)
    best_deviation_strategy = None

    for team1_actions in itertools.product(range(num_actions_per_player), repeat=num_players_per_team):
        deviation_strategy = list(team1_actions) + [equilibrium[i] for i in range(num_players_per_team, num_players_per_team*2)]
        deviation_strategy = convert(deviation_strategy, num_actions_per_player)
        deviation_payoff = calculate_expected_payoffs(num_players_per_team*2, num_actions_per_player, payoff, deviation_strategy)

        if np.sum(deviation_payoff[:num_players_per_team]) > np.sum(best_deviation_payoff[:num_players_per_team]):
            best_deviation_payoff = deviation_payoff
            best_deviation_strategy = deviation_strategy

    return best_deviation_payoff, best_deviation_strategy

def generate_mixed_strategies(num_actions, step=0.0001):
    if num_actions == 1:
        return [{0: 1.0}]
    
    strategies = []
    prob = 0.0
    while prob <= 1.0:
        strategy = {0: prob, 1: 1.0 - prob}
        strategies.append(strategy)
        prob += step
    return strategies

def calculate_mixed_cooperative_deviation_payoffs(num_players_per_team, num_actions_per_player, payoff_tensor, equilibrium):
    best_deviation_payoff = np.zeros(num_players_per_team*2)
    best_deviation_strategy = None

    team1_mixed_strategies = [generate_mixed_strategies(num_actions_per_player, step=0.1) for _ in range(num_players_per_team)]
    for team1_strategy_combination in itertools.product(*team1_mixed_strategies):
        deviation_strategy = list(team1_strategy_combination) + [equilibrium[i] for i in range(num_players_per_team, num_players_per_team*2)]
        deviation_payoff = calculate_expected_payoffs(num_players_per_team*2, num_actions_per_player, payoff_tensor, deviation_strategy)

        if np.sum(deviation_payoff[:num_players_per_team]) > np.sum(best_deviation_payoff[:num_players_per_team]):
            best_deviation_payoff = deviation_payoff
            best_deviation_strategy = deviation_strategy

    return best_deviation_payoff, best_deviation_strategy

# convert to the same format
def convert(ne, num_actions_per_player):
    mixed_strategy = []
    for player_strategy in ne:
        strategy_dict = {}
        for action in range(num_actions_per_player):
            strategy_dict[action] = player_strategy.get(action, 0) if isinstance(player_strategy, dict) else (1.0 if action == player_strategy else 0.0)
        mixed_strategy.append(strategy_dict)
    return mixed_strategy

if __name__ == "__main__":
    seed = 3
    num_players_per_team = 2
    num_actions_per_player = 2
    payoff = generate_team_payoff(num_players_per_team, num_actions_per_player, seed)
    print(f"Generated payoff is:")
    print(payoff)
    
    game = Game(payoff)
    ne = game.find_all_equilibria()
    print(f"Find {len(ne)} Nash equilibria!")
    
    for i in range(len(ne)):  
        converted_ne = convert(ne[i], num_actions_per_player)
        print(f"The {i+1}th NE is {converted_ne}")
        expected_payoff = calculate_expected_payoffs(num_players_per_team*2, num_actions_per_player, payoff, converted_ne)
        print(f"Expected payoff for {i+1}th NE is {expected_payoff}")
        
        cooperative_deviation_payoff, deviation_strategy = calculate_pure_cooperative_deviation_payoffs(num_players_per_team, num_actions_per_player, payoff, ne[i])
        print(f"Best cooperative deviation pure strategy is {deviation_strategy}")
        print(f"Best cooperative deviation pure payoff for {i+1}th NE is {cooperative_deviation_payoff}")
        
        cooperative_mixed_deviation_payoff, deviation_mixed_strategy = calculate_mixed_cooperative_deviation_payoffs(num_players_per_team, num_actions_per_player, payoff, ne[i])
        print(f"Best cooperative deviation mixed strategy is {deviation_mixed_strategy}")
        print(f"Best cooperative deviation mixed payoff for {i+1}th NE is {cooperative_mixed_deviation_payoff}")