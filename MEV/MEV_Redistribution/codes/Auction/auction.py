def initialize(num_builders, num_searchers, builder_utilities, searcher_utilities):
    # Initialize all builders to have a utility of 0
    for builder in range(1, num_builders + 1):
        builder_utilities[builder] = 0
    
    #Intialize all searchers to have a utility of 0
    for searcher in range(1, num_searchers + 1):
        searcher_utilities[searcher] = 0
    
    return searcher_utilities, builder_utilities

def compute_searcher_utilities(winners, searcher_valuations, searcher_utilities):
    # Iterate through each OFA and find the searcher with the highest valuation
    for ofa, ofa_bids in searcher_valuations.items():
        # Find the highest bid
        max_valuation = max(ofa_bids)
        
        # Find the index of the winning searcher
        winning_searcher = ofa_bids.index(max_valuation)
        ofa_bids.pop(winning_searcher)

        # Find the second-highest bid that the winning searcher has to pay
        if len(ofa_bids) == 0:
            bid = 0
        else:
            bid = max(ofa_bids)

        # Insert the maximum bid back into the list
        ofa_bids.insert(winning_searcher, max_valuation)
        
        # Map searcher to the OFA they have won
        if winning_searcher not in winners:
            winners[winning_searcher] = [ofa]
        else:
            winners[winning_searcher].append(ofa)

        # Calculate utility and store it in the searcher_utilities dictionary
        utility = max_valuation - bid
        searcher_utilities[winning_searcher] += utility
        
        print(f"The Searcher {winning_searcher} wins OFA {ofa} with a utility of {utility}")
    return winners, searcher_utilities

def compute_builder_utilities(winners, builder_valuations, builder_utilities):
    # Loop through each winning Searcher's bundle
    for searcher, ofas_won in winners.items():
        for ofa in ofas_won:
            # Increment the utility of the builder for every bundle searcher gives him
            builder_utilities[1] += builder_valuations[1][searcher]
    return builder_utilities

def utility_computation(num_builders, num_searchers, num_ofas, builder_valuations, searcher_valuations):
    # Initialize dictionaries to store the utilities of each searcher and builder respectively
    searcher_utilities, builder_utilities = initialize(num_builders,num_searchers, {}, {})

    # winners is a dictionary to store all searchers who have won atleast one OFA
    # and the OFAs they have won respectively
    # searcher_utilities is a dictionary to store the utility of each searcher
    winners, searcher_utilities = compute_searcher_utilities({}, searcher_valuations, searcher_utilities)

    # builder_utilities is a dictionary to store the utility of each builder
    builder_utilities = compute_builder_utilities(winners, builder_valuations, builder_utilities)
    
    return searcher_utilities, builder_utilities