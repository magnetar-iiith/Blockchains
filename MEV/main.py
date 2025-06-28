from Auction.auction import utility_computation
from Coalition.coalition import calculate_shapley_value

def auction_input():
    # Get the number of Builders, Searchers, and OFAs from the user
    num_builders = int(input("Enter the number of Builders: "))
    num_searchers = int(input("Enter the number of Searchers: "))
    num_ofas = int(input("Enter the number of OFAs: "))

    searcher_valuations = input_searcher_valuations({}, num_ofas, num_searchers)

    builder_valuations = input_builder_valuations({}, num_builders, num_searchers)
    
    searcher_utilities, builder_utilities = utility_computation(num_builders, num_searchers, num_ofas, builder_valuations, searcher_valuations)
    
    print(searcher_utilities, builder_utilities)

def input_searcher_valuations(searcher_valuations, num_ofas, num_searchers):
    # Loop through each OFA
    for ofa in range(1, num_ofas + 1):
        # Initialize a list to store searcher valuations (which is equal to their bid) for this OFA
        ofa_bids = []
        
        # Loop through each searcher in this OFA
        for searcher in range(1, num_searchers + 1):
            valuation = float(input(f"Enter the valuation of Searcher {searcher} in OFA {ofa}: "))
            ofa_bids.append(valuation)
        
        # Store the list of searcher valuations for this OFA in the dictionary
        searcher_valuations[ofa] = ofa_bids
    return searcher_valuations

def input_builder_valuations(builder_valuations, num_builders, num_searchers):
    # Loop through each builder
    for builder in range(1, num_builders + 1):
        # Initialize a list to store valuations for each searcher with this builder
        searcher_valuations = []
        
        # Loop through each searcher
        for searcher in range(1, num_searchers + 1):
            valuation = float(input(f"Enter the valuation of Builder {builder} for Searcher {searcher}: "))
            searcher_valuations.append(valuation)
        
        # Store the list of valuations for this builder in the dictionary
        builder_valuations[builder] = searcher_valuations
    return builder_valuations

def are_entries_zero(tuple_data, i, j):
    # Check if all entries from index i to index j are 0
    for index in range(i, j + 1):
        if tuple_data[index] != 0:
            return False
    return True

def initialize_tuple(tuple_data, num_builders, num_searchers, num_ofas):
    set = ("P",)
    for i in range(1, num_builders + 1):
        if tuple_data[i] == 1:
            set += (f"B{i}",)
    for i in range(num_builders + 1, num_builders + num_searchers + 1):
        if tuple_data[i] == 1:
            set += (f"S{i - num_builders}",)
    for i in range(num_builders + num_searchers + 1, num_builders + num_searchers + num_ofas + 1):
        if tuple_data[i] == 1:
            set += (f"OFA{i - num_builders - num_searchers}",)
    return set

def exactly_one_builder(tuple_data, num_builders):
    # Check if exactly one builder is in the coalition
    count = 0
    for i in range(1, num_builders + 1):
        if tuple_data[i] == 1:
            count += 1
    if count == 1:
        return True
    return False

def multiple_builders(utility_dict, coalition, num_builders, num_searchers, num_ofas):
    num_players = num_builders + num_searchers + num_ofas + 1
    utility = 0
    proposer = (1,)
    builder = tuple(0 for _ in range(1, num_builders + 1))
    rest = coalition[num_builders + 1:]
    for idx, bit in enumerate(coalition[1: num_builders + 1]):
        if bit == 1:
            single_builder = proposer + builder[:idx] + (1,) + builder[idx + 1:] + rest
            utility = max(utility, utility_dict[single_builder])
    utility_dict[coalition] = utility
    return utility_dict

def coalition_input():
    try:
        # Get the number of Builders from the user
        num_builders = int(input("Enter the number of Builders: "))
    except ValueError:
        print("Please enter an integral value for the number of Builders.")
    try:
        # Get the number of Searchers from the user
        num_searchers = int(input("Enter the number of Searchers: "))
    except ValueError:
        print("Please enter an integral value for the number of Searchers.")
    try:
        # Get the number of OFAs from the user
        num_ofas = int(input("Enter the number of OFAs: "))
    except ValueError:
        print("Please enter an integral value for the number of OFAs.")
    
    # Calculate the number of players
    num_players = num_builders + num_searchers + num_ofas + 1 # Including proposer

    # Initialize a dictionary to store the utility of each coalition
    utility_dict = {}

    for i in range(2**num_players):
        coalition = tuple(int(bit) for bit in bin(i)[2:].zfill(num_players))
        if coalition[0] == 0:
            utility_dict[coalition] = 0
        elif are_entries_zero(coalition, 1, num_builders):
            utility_dict[coalition] = 0
        elif are_entries_zero(coalition, num_builders + 1, num_builders + num_searchers):
            utility_dict[coalition] = 0
        elif are_entries_zero(coalition, num_builders + num_searchers + 1, num_builders + num_searchers + num_ofas):
            utility_dict[coalition] = 0
        elif exactly_one_builder(coalition, num_builders):
            set = initialize_tuple(coalition, num_builders, num_searchers, num_ofas)
            utility = float(input(f"Enter the utility for coalition {set}: "))
            utility_dict[coalition] = utility
        else:
            utility_dict = multiple_builders(utility_dict, coalition, num_builders, num_searchers, num_ofas)
    
    # # Print the resulting dictionary
    # print("Utility Dictionary:")
    # for coalition, utility in utility_dict.items():
    #     print(f"{coalition}: {utility}")
    
    # Calculate Shapley values
    shapley_values = calculate_shapley_value(utility_dict, num_builders, num_searchers, num_ofas)
    print(shapley_values)

while(True):
    print("Choose one of the following options (enter corresponding numbers):")
    print("1. Auction")
    print("2. Coalition")
    print("3. Exit")
    try:
        option = int(input())
        if option == 1:
            auction_input()
        elif option == 2:
            coalition_input()
        elif option == 3:
            break
    except ValueError:
        print("Please enter either 1, 2 or 3.")
