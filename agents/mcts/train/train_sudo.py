from agents.mcts.mcts_node import MCTNode

if __name__ == '__main__':
    epoch = 1000
    for i in range(epoch):
        tree = MCTNode().load()
        game_count = 5000
        episodes = []
        while game_count:
            p_network,v_network = net() #static
            tree.select() # using the v and p network, stores a list of transitions till the leaf node
            tree.expend() # choce a random move, then observe state of move, create state node
            tree.simualte(s) # called in the expand, return a simulated games,
            # outcome value. Play the game from the state s (can take fast rollout policy)
            # inside the simulate()
                agent0 = PolicyAgent() # p_networkd
                agent1 = PolicyAgent() # p_networkd
                game = Mangala(state, agent0, agent1)
                outcome = game.start() # if win, return 1, if lose, return -1
            tree.backup(outcome) # propagate the value back to the root node w and n,
            # does not need the history in the rollout
            episodes.append(tree.get_tracjectory(expand_state)) # get the episode from the root node
            game_count -=1
        tree.save()
        network.train(episodes)






