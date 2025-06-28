import itertools

def initialize_tuple(tup, num_players):
    t = tuple(0 for _ in range(num_players))
    for idx in tup:
        t = t[:idx] + (1,) + t[idx + 1:]
    return t

def calculate_shapley_value(utility_dict, num_builders, num_searchers, num_ofas):
    num_players = num_builders + num_searchers + num_ofas + 1
    fact = [1] * (num_players + 1)
    factorial(num_players, fact)
    shapley_values = [0] * num_players
    for perm in itertools.permutations(range(num_players), num_players):
        for player in range(num_players):
            i = perm.index(player)
            with_player = initialize_tuple(perm[:i + 1], num_players)
            without_player = initialize_tuple(perm[:i], num_players)
            marginal_value = utility_dict[with_player] - utility_dict[without_player]
            shapley_values[player] += marginal_value

    total_players = fact[num_players]
    shapley_values = [value / total_players for value in shapley_values]

    return shapley_values

def factorial(n, fact):
    if n == 0:
        return 1
    fact[n] = n * factorial(n - 1, fact)
    return fact[n]
