""" Lily Xu

implement MINION algorithm from AAMAS 2019 paper
"""

import os, sys
import argparse
import numpy as np
import pandas as pd
import gurobi as grb

from sklearn.tree import DecisionTreeClassifier
from scipy.stats import bernoulli

import matplotlib.pyplot as plt

from generate_graph import *


SHOW_RUNTIME = False
VERBOSE = False
out_path = './plots'

if not os.path.exists(out_path):
    os.makedirs(out_path)


def generate_graph(width, height):
    """ Create rectangular graph of specified dimensions

    :param width
    :param height

    :return dict of nodes and adjacency lists
    """

    assert isinstance(width, int) and isinstance(height, int)
    num_targets = width * height

    graph = {}
    id_to_coord = {}
    coord_to_id = {}

    def get_neighbors(i, j):
        neighbors = []
        if i > 0:
            neighbors.append((i-1, j))
        if j > 0:
            neighbors.append((i, j-1))
        if i < width - 1:
            neighbors.append((i+1, j))
        if j < height - 1:
            neighbors.append((i, j+1))

        # assume we can stay at the same cell for multiple rounds
        neighbors.append((i, j))

        return neighbors

    # create nodes
    id = 0
    for i in range(width):
        for j in range(height):
            id_to_coord[id] = (i,j)  # store coordinates for each target id
            coord_to_id[(i,j)] = id
            graph[id] = set()
            id += 1

    # set neighbors
    id = 0
    for i in range(width):
        for j in range(height):
            # go through neighbors
            neighbors = get_neighbors(i, j)
            for neighbor in neighbors:
                graph[id].add(coord_to_id[neighbor])
            id += 1

    return graph, id_to_coord, coord_to_id



class Minion:
    def __init__(self, width, height, D, horizon, M, patrol_post=1,
                attack_prob=None, lamb=None, w1=None, w2=None, w3=None):
        """ Create Minion class

            (for QR or SUQR adversary)
            :param lamb - lambda. measure of rationality
                                  - zero = perfectly irrational
                                  - higher = closer to perfect rationality
            :param def_mixed_strategy
                                  - NOTE: adversary's view of our mixed strategy is
                                    **all** our historical actions

            (for SUQR adversary)
            :param w1, w2, w3 (float) - weights for SU function
            """

        self.D       = D                # number of rounds of game
        self.horizon = horizon          # patrol length
        self.start   = patrol_post
        self.end     = patrol_post
        self.M       = M                # number of attacks
        self.num_loc = width * height

        self.width  = width
        self.height = height

        graph, id_to_coord, coord_to_id = generate_graph(width, height)
        self.graph   = graph
        self.id_to_coord = id_to_coord
        self.coord_to_id = coord_to_id

        # ground truth adversary attack probability
        # for stationary adversary
        assert np.isclose(attack_prob.sum(), 1)
        assert len(attack_prob) == self.num_loc
        self.attack_prob = attack_prob

        # used for QR and SUQR model
        # note: lambda that experts suggest is 0.91 (Nguyen 2013)
        self.lamb = lamb

        # SUQR model
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

        # utility (payoff for each attacked target)
        self.U_adv_c = None  # adversary, covered
        self.U_adv_u = None  # adversary, uncovered
        self.U_def_c = None  # defender, covered
        self.U_def_u = None  # defender, uncovered

        self.features = None


    def set_historical_data(self, num_timesteps):
        """ Simulate historical attacks and generate data

        :param num_timesteps
        """

        attacks = self.get_attacks(num_timesteps)
        self.historical_data = attacks


    def get_time_unrolled_graph(self, start, end, start_t=0, end_t=None):
        """ Convert graph to time-unrolled graph
        with specified start and end nodes

        :param start (int) - location of start node
        :param end (int) - location of goal node
        :param start_t (int) - time of start
        :param end_t (int) - time of end

        :return dict of time-unrolled graph
                - key: (location, time)
                - val: set of neighbors
        """
        assert start in self.graph and end in self.graph
        if end_t == None: end_t = self.horizon - 1
        assert 0 <= start_t < self.horizon - 1
        assert 0 < end_t <= self.horizon - 1

        tu_graph = {} # time-unrolled graph

        # add start node
        tu_graph[(start, start_t)] = set()
        for neighbor in self.graph[start]:
            tu_graph[(start, start_t)].add((neighbor, start_t+1))

        for t in range(start_t+1, end_t):
            for l in self.graph.keys():
                # add node
                tu_graph[(l, t)] = set()

                if t == end_t - 1:
                    # only add edges to the goal node
                    if end in self.graph[l]:
                        tu_graph[(l, t)].add((end, end_t))
                    continue

                for neighbor in self.graph[l]:
                    tu_graph[(l, t)].add((neighbor, t+1))

        # add end node
        tu_graph[(end, end_t)] = set()

        if VERBOSE:
            print('time unrolled graph')
            for l in tu_graph:
                print(' {}: {}'.format(l, tu_graph[l]))

        return tu_graph


    def strategy_to_attacks(self, adv_strategy):
        """ Simulate adversary behavior at each round
        """

        assert adv_strategy.shape[0] == self.num_loc

        # adv_strategy may be a one-dimensional array instead of 2D
        if adv_strategy.ndim == 1:
            prob = adv_strategy / adv_strategy.sum()
            attacks = np.zeros(self.num_loc)
            attack_l = np.random.choice(list(self.graph.keys()), self.M,
                                    p=prob)
            loc, num_attacks = np.unique(attack_l, return_counts=True)
            attacks[loc] = num_attacks

        else:
            num_rounds = adv_strategy.shape[1]
            attacks = np.zeros((self.num_loc, num_rounds))

            # select random attacks based on attack probability
            for d in range(num_rounds):
                prob = adv_strategy[:, d] / adv_strategy[:, d].sum()
                attack_l = np.random.choice(list(self.graph.keys()), self.M,
                                        p=prob)

                loc, num_attacks = np.unique(attack_l, return_counts=True)

                attacks[loc, d] = num_attacks

        return attacks


    def get_attacks(self, num_timesteps=None):
        """ Simulate adversary behavior in a single round

        adversary selects M distinct (l, t) pairs based on attack probabilities

        :return (L, T) array with 1 in locations of attack
        """

        if num_timesteps is None: num_timesteps = self.horizon

        # select random attacks based on attack probability
        attack_l = np.random.choice(list(self.graph.keys()), self.M,
                                    p=self.attack_prob)
        attack_t = np.random.randint(num_timesteps, size=self.M)

        attacks = np.zeros((self.num_loc, num_timesteps))
        attacks[attack_l, attack_t] = 1
        return attacks


    def get_attacks_rounds(self, num_rounds=None):
        """ Simulate adversary behavior across multiple rounds

        adversary selects M distinct (l, t) pairs based on attack probabilities

        :return (L, num_rounds) array with 1 in locations of attack
        """

        if num_rounds is None: num_rounds = self.D

        attacks = np.zeros((self.num_loc, num_rounds))

        # select random attacks based on attack probability
        for d in range(num_rounds):
            attack_l = np.random.choice(list(self.graph.keys()), self.M,
                                    p=self.attack_prob)

            loc, num_attacks = np.unique(attack_l, return_counts=True)

            attacks[loc, d] = num_attacks

        return attacks


    def get_defender_strategy(self, method,
                              start=None, end=None, start_t=None, end_t=None,
                              est_attack_prob=None,
                              alpha=None, est_reward=None, z=None,
                              adv_actions=None):
        """ AAMAS 2019 mathematical program to compute defender strategy

        :param method (str) - either {'ml', 'ol', 'hindsight'}
        :param start (int) - start node
        :param end (int) - end node
        :param start_t (int) - start timestep
        :param end_t (int) - end timestep

        ------------------------------------------
        AAMAS 2019 mathematical program 3
        mathematical model Q(start, end)

        (for ML)
        :param est_attack_prob

        ------------------------------------------
        AAMAS 2019 mathematical program 2
        mathematical model P(start, end)

        (for OL)
        :param alpha
        :param est_reward
        :param z

        ----------

        (for hindsight optimal)
        :param adv_actions

        :return def_strategy (L, T)
        """

        if start is None:
            start = self.start
            start_t = 0
        else:
            assert start in self.graph
            assert 0 <= start_t < self.horizon - 1
            if start_t == 0:
                assert start == self.start

        if end is None:
            end = self.end
            end_t = self.horizon - 1
        else:
            assert end in self.graph
            assert 0 < end_t <= self.horizon - 1
            if end_t == self.horizon - 1:
                assert end == self.end

        assert isinstance(start_t, int) and isinstance(end_t, int)
        assert method in {'ol', 'ml', 'hindsight'}
        if method == 'ol':
            assert alpha is not None and est_reward is not None and z is not None
            assert alpha == 0 or alpha == 1
            assert est_reward.shape == (self.num_loc,)
            assert z.shape == (self.num_loc, self.horizon)
        elif method == 'ml':
            assert est_attack_prob is not None
            assert est_attack_prob.shape == (self.num_loc,)
            assert np.isclose(est_attack_prob.sum(), 1)
        elif method == 'hindsight':
            assert adv_actions is not None

        # no timesteps between start and end
        if end_t - start_t == 1:
            # infeasible
            if end not in self.graph[start]:
                return None

            else:
                def_strategy = np.zeros((len(self.graph.keys()), self.horizon))

                def_strategy[start, start_t] = 1
                def_strategy[end, end_t] = 1

                return def_strategy

        assert end_t - start_t > 1

        if VERBOSE:
            print('start ({}, {}) end ({}, {})'.format(start, start_t,
                                                       end, end_t))

        tu_graph = self.get_time_unrolled_graph(start, end, start_t, end_t)

        set_L = list(self.graph.keys()) # locations
        set_T = range(start_t + 1, end_t) # time - don't include source and sink
        set_E = set([(k, v) for k in tu_graph for v in tu_graph[k]])  # edges

        # get in-neighbors for the time-unrolled graph
        def get_in_neighbors(l, t):
            if t == start_t + 1:
                # special case: at the beginning, from the source
                if l in self.graph[start]:
                    return set([start])
                else:
                    return set()
            else:
                return self.graph[l]

        # get out-neighbors for the time-unrolled graph
        def get_out_neighbors(l, t):
            if t == end_t - 1:
                # special case: just before the end time, going into the sink
                if end in self.graph[l]:
                    return set([end])
                else:
                    return set()
            else:
                return self.graph[l]


        opt_model = grb.Model(name='MIP defender {} method'.format(method))

        # suppress output
        if not VERBOSE:
            opt_model.setParam('OutputFlag', 0)


        ######################
        # set up variables
        ######################
        # binary random variable x = v_d
        x_vars = {(l, t): opt_model.addVar(vtype=grb.GRB.BINARY,
                                name='x_{}_{}'.format(l, t))
        for t in set_T for l in set_L}

        # network flow
        f_vars = {(e): opt_model.addVar(vtype=grb.GRB.BINARY,
                                name='f_{}'.format(e))
        for e in set_E}


        ######################
        # set up constraints
        ######################
        for t in set_T:
            for l in set_L:
                # defender strategy equal to flow in node
                opt_model.addConstr(
                    lhs=x_vars[l, t],
                    sense=grb.GRB.EQUAL,
                    rhs=grb.quicksum(f_vars[((l, t), (neighbor, t+1))]
                                     for neighbor in get_out_neighbors(l, t)),
                    name='visit_location_{}_{}'.format(l, t))

                # flow in equals flow out
                opt_model.addConstr(
                    lhs=grb.quicksum(f_vars[((neighbor, t-1), (l, t))]
                                     for neighbor in get_in_neighbors(l, t)),
                    sense=grb.GRB.EQUAL,
                    rhs=grb.quicksum(f_vars[((l, t), (neighbor, t+1))]
                                     for neighbor in get_out_neighbors(l, t)),
                    name='equality_{}_{}'.format(l, t))

        # source has out-degree 1
        opt_model.addConstr(
            lhs=grb.quicksum(f_vars[((start, start_t), (neighbor, start_t + 1))]
                             for neighbor in self.graph[start]),
            sense=grb.GRB.EQUAL,
            rhs=1,
            name='souce_out')

        # sink has in-degree 1
        opt_model.addConstr(
            lhs=grb.quicksum(f_vars[((neighbor, end_t - 1), (end, end_t))]
                             for neighbor in self.graph[end]),
            sense=grb.GRB.EQUAL,
            rhs=1,
            name='sink_in')


        ######################
        # set up objective
        ######################
        if method == 'ml':
            objective = grb.quicksum(x_vars[l, t] * est_attack_prob[l]
                                     for t in set_T for l in set_L)
        elif method == 'ol':
            objective = grb.quicksum(x_vars[l, t] * (alpha * est_reward[l] + z[l, t])
                                     for t in set_T for l in set_L)

        elif method == 'hindsight':
            objective = grb.quicksum(x_vars[l, t] * np.sum(adv_actions[l, :])
                                     for t in set_T for l in set_L)

        # set objective to maximize
        opt_model.ModelSense = grb.GRB.MAXIMIZE
        opt_model.setObjective(objective)

        opt_model.optimize()

        # other status codes: INFEASIBLE, INF_OR_UNBD, UNBOUNDED, etc
        if opt_model.Status != grb.GRB.OPTIMAL:
            # model was not solved to optimality
            return None


        ######################
        # evaluate
        ######################

        opt_df = pd.DataFrame.from_dict(x_vars, orient='index', columns=['var'])
        opt_df.index = pd.MultiIndex.from_tuples(opt_df.index, names=['l', 't'])
        opt_df.reset_index(inplace=True)
        opt_df['solution_value'] = opt_df['var'].apply(lambda item: item.X)

        def_strategy = np.zeros((len(self.graph.keys()), self.horizon))
        def_strategy[opt_df['l'], opt_df['t']] =  opt_df['solution_value']

        # add start and end positions
        def_strategy[start, start_t] = 1
        def_strategy[end, end_t] = 1

        if VERBOSE:
            print('optimal solution\n', opt_df)
            print('defender strategy\n', def_strategy)

        return def_strategy


    def get_defender_utility(self, attacks, def_strategy):
        """
        :param attacks (L,)
        :param def_strategy (L,)

        :return value of defender utility
        """
        # def_payoff_c = np.tile(self.U_def_c, (def_strategy.shape[1], 1)).T
        # def_payoff_u = np.tile(self.U_def_u, (def_strategy.shape[1], 1)).T

        utility = np.sum(attacks * def_strategy * self.U_def_c) + \
                  np.sum(attacks * (1 - def_strategy) * self.U_def_u)

        return utility


    def adversary_response(self, adv_model,
                            def_mixed_strategy=None):
        """ Compute adversary response based on behavior model

        Model the probability that the adversary chooses each target

        :param adv_model (str) stc - stochastic, (stationary)
                           qr - (nonstationary)
                           suqr - (nonstationary)

        (for stochastic adversary)
        :param attack_prob (array)

        (for QR or SUQR adversary)
        :param lamb - lambda. measure of rationality
                              - zero = perfectly irrational
                              - higher = closer to perfect rationality
        :param def_mixed_strategy
                              - NOTE: adversary's view of our mixed strategy is
                                **all** our historical actions

        (for SUQR adversary)
        :param w1, w2, w3 (float) - weights for SU function

        :return array (L, T)
        """
        self.check_params(adv_model)


        if adv_model == 'stc':
            adv_response = self.M * self.attack_prob

        else:
            # QR utility
            if adv_model == 'qr':
                adv_utility = self.U_adv_u * (1 - def_mixed_strategy) + \
                              self.U_adv_c * def_mixed_strategy

            # adversary SUQR utility
            # subjective utility function
            elif adv_model == 'suqr':
                adv_utility = self.w1 * def_mixed_strategy + self.w2 * self.U_adv_u + \
                              self.w3 * self.U_adv_c

            adv_response = self.M * np.exp(self.lamb * adv_utility) / \
                           np.exp(self.lamb * adv_utility).sum()

        return adv_response


    def compute_hindsight_regret(self, rewards, adv_actions):
        """ Compute optimal hindsight strategy and regret

        :param rewards (D,) is the rewards we received in each round
        :param adv_actions (L, D) are the actual attacks from each *round*


        :return (D,) average hindsight regret at each round
        """
        assert adv_actions.shape == (self.num_loc, self.D)
        assert len(rewards) == self.D

        optimal_utility = np.zeros(self.D)

        for d in range(self.D):
            def_strategy = self.get_defender_strategy(method='hindsight',
                                adv_actions=adv_actions[:, :d+1])

            attacks_flat = np.sum(adv_actions[:, :d+1], axis=1)
            def_strategy_flat = np.sum(def_strategy, axis=1)

            optimal_utility[d] = self.get_defender_utility(
                                    attacks_flat, def_strategy_flat)

        # cumulative regret with each round
        cum_regrets = optimal_utility - np.cumsum(rewards)

        # average regret per round
        avg_regret = cum_regrets / np.arange(1, self.D + 1)

        return avg_regret


    def geometric_resampling(self, gamma, eta, w, est_reward):
        """ Algorithm 2: GR algorithm

        from Neu and Bartok, ALT 2013

        :param gamma (float) probability of exploration
        :param eta (float)
        :param w
        :param est_reward

        :return (L, T) matrix
        """

        assert est_reward.shape == (self.num_loc,)

        # initialize
        K = np.zeros((self.num_loc, self.horizon))
        for k in range(w):
            # execute steps 3-13 of algorithm 1
            est_v = self.online_learner_helper(gamma, eta, est_reward)

            for l in range(self.num_loc):
                for t in range(self.horizon):
                    if K[l, t] != 0:
                        continue

                    if k < w and est_v[l, t] == 1:
                        K[l, t] = k
                    elif k == w:
                        K[l, t] = w

            if np.all(K > 0): break

        return K


    def online_learner_helper(self, gamma, eta, est_reward):
        """ Lines 3 -- 13 of Algorithm 1: MINION-sm

        also called by MINION and geometric resampling

        :param gamma: flag=0 with probability gamma
        :param eta
        :param est_reward

        :return def_strategy
        """
        # sample flag
        flag = bernoulli.rvs(1 - gamma)

        z = np.random.exponential(eta, (self.num_loc, self.horizon))

        if flag == 0:
            # exploration
            if VERBOSE:
                print('\n\nexplore, flag = {}, gamma = {}'.format(flag, gamma))

            alpha = 0

            num_repeats = 0
            max_repeats = 10
            # repeat until we have a valid path
            while True:
                # select random timestep
                j_t = np.random.randint(1, self.horizon - 1)

                # select random target from those that are valid
                valid_targets = set()
                max_dist = min(j_t, self.horizon - j_t)

                while True:
                    j_l = np.random.randint(self.num_loc)

                    # avoid selecting patrol post as random target
                    if j_l == self.start:
                        continue

                    # if we're at the first or last timestep,
                    # ensure we can get to the patrol post
                    if j_t == 1:
                        if j_l in self.graph[self.start]: break
                    elif j_t == self.horizon - 1:
                        if self.end in self.graph[j_l]: break

                    # ensure euclidean distance to patrol post is within time allowed
                    tup_l = self.id_to_coord[j_l]
                    tup_s = self.id_to_coord[self.start]
                    tup_e = self.id_to_coord[self.end]
                    dist1 = np.abs(tup_l[0] - tup_s[0]) + \
                            np.abs(tup_l[1] - tup_s[1])
                    dist2 = np.abs(tup_l[0] - tup_e[0]) + \
                            np.abs(tup_l[1] - tup_e[1])
                    if (dist1 <= j_t) and (dist2 <= self.horizon - j_t):
                        break

                if VERBOSE:
                    print('  random target: j = ({}, {}) after {} repeats'.
                                format(j_l, j_t, num_repeats))

                strategy1 = self.get_defender_strategy(method='ol',
                            start=self.start, end=j_l, start_t=0, end_t=j_t,
                            alpha=alpha, est_reward=est_reward, z=z)

                strategy2 = self.get_defender_strategy(method='ol',
                            start=j_l, end=self.end,
                            start_t=j_t, end_t=self.horizon-1,
                            alpha=alpha, est_reward=est_reward, z=z)

                # ensure strategy is valid
                if strategy1 is not None and strategy2 is not None: break

                num_repeats += 1
                if num_repeats > max_repeats:
                    raise Exception('Exceeded max_repeats={} to find a valid random path in online learner.'.format(max_repeats))

            # combine two halves of strategy
            def_strategy = strategy1 + strategy2
            # ensure the overlap is not being double counted
            def_strategy[def_strategy > 1] = 1
        else:
            # exploitation
            if VERBOSE:
                print('\n\nexploit, flag = {}, gamma = {}'.format(flag, gamma))
                print('estimated reward', est_reward)
            alpha = 1
            def_strategy = self.get_defender_strategy(method='ol',
                        alpha=alpha, est_reward=est_reward, z=z)

        if VERBOSE:
            print('  defender strategy: \n', def_strategy)

        return def_strategy


    def online_learner(self, adv_model, eta, w, gamma):
        """ Algorithm 1: MINION-sm algorithm (online learner)

        :param eta (float)
        :param w (int) passed to geometric resampling
        :param gamma (float) probability of exploration

        :param adv_model (str) {'stc', 'qr', 'suqr'}
        """

        if VERBOSE:
            print()
            print('-------------------')
            print('online learner')
            print('-------------------')

        assert eta > 0
        assert isinstance(w, int)
        assert w > 0
        assert 0 <= gamma <= 1

        self.check_params(adv_model)

        adv_mixed_strategy = np.zeros((self.num_loc, self.D))
        all_attacks        = np.zeros((self.num_loc, self.D))

        # track all defender actions, which is what the adversary observes as mixed strategy
        all_def_strategy   = np.zeros((self.num_loc, self.D))

        all_est_reward     = np.zeros((self.num_loc, self.D))

        # initialize estimated reward
        # NOTE: assuming reward is constant across timesteps
        est_reward         = np.zeros(self.num_loc)
        rewards            = np.zeros(self.D)

        for d in range(self.D):
            # learn defender strategy
            def_strategy = self.online_learner_helper(gamma, eta, est_reward)

            def_strategy = np.sum(def_strategy, axis=1)
            all_def_strategy[:, d] = def_strategy
            all_est_reward[:, d] = est_reward

            # adversary pick targets
            if adv_model == 'stc':
                adv_mixed_strategy[:, d] = self.adversary_response('stc')

            elif adv_model == 'qr':
                cum_def_actions = np.sum(all_def_strategy[:, :d+1], axis=1)
                adv_mixed_strategy[:, d] = self.adversary_response('qr',
                def_mixed_strategy=cum_def_actions)

            elif adv_model == 'suqr':
                cum_def_actions = np.sum(all_def_strategy[:, :d+1], axis=1)
                adv_mixed_strategy[:, d] = self.adversary_response('suqr',
                                    def_mixed_strategy=cum_def_actions)

            attacks = self.strategy_to_attacks(adv_mixed_strategy[:, d])
            all_attacks[:, d] = attacks

            # defender play strategy
            K = self.geometric_resampling(gamma, eta, w, est_reward)
            K = np.sum(K, axis=1)

            # est_reward += np.sum(K * attacks * def_strategy, axis=1)
            est_reward += K * attacks * def_strategy

            rewards[d] = self.get_defender_utility(attacks, def_strategy)

        regrets = self.compute_hindsight_regret(rewards, all_attacks)

        return {'rewards': rewards, 'regrets': regrets, 'attacks': all_attacks,
                'all_def_strategy': all_def_strategy, 'all_est_reward': all_est_reward}


    def hybrid(self, adv_model, eta, beta, w, gamma, e):
        """ Algorithm 3: MINION algorithm (hybrid model)

        :param adv_model

        :return reward, adv_respones
        """

        if VERBOSE:
            print()
            print('-------------------')
            print('hybrid model')
            print('-------------------')

        assert eta > 0
        assert beta > 0
        assert isinstance(w, int) and w > 0
        assert 0 <= gamma <= 1
        assert isinstance(e, int) and e >= 0

        self.check_params(adv_model)

        reward_ml = 0
        reward_ol = 0
        num_ml = 0
        num_ol = 0
        rewards = np.zeros(self.D)

        # initialize estimated reward
        est_reward = np.zeros(self.num_loc)

        predictions = self.ml_predictions()
        est_attack_prob = predictions / predictions.sum()

        all_def_strategy = np.zeros((self.num_loc, self.D))
        all_adv_strategy = np.zeros((self.num_loc, self.D))
        all_attacks      = np.zeros((self.num_loc, self.D))
        all_est_reward   = np.zeros((self.num_loc, self.D))

        arm_pulls = []

        for d in range(self.D):
            # ensure that ML and OL each get pulled at least once
            # to avoid division by zero
            if d == 0:
                flag1 = True
            elif d == 1:
                flag1 = False
            # explore uniformly between ML and OL to initialize rewards
            elif d < e:
                flag1 = np.random.choice([True, False])
            else:
                c1 = np.random.exponential(beta)
                c2 = np.random.exponential(beta)
                flag1 = (reward_ml / num_ml + c1) < (reward_ol / num_ol + c2)

            if flag1:
                # use online model - algorithm 1
                def_strategy = self.online_learner_helper(gamma, eta, est_reward)
            else:
                # use ML model - from LP Q
                def_strategy = self.get_defender_strategy(method='ml',
                                    est_attack_prob=est_attack_prob)

            def_strategy = np.sum(def_strategy, axis=1)
            all_def_strategy[:, d] = def_strategy

            # adversary pick targets
            if adv_model == 'stc':
                adv_strategy = self.adversary_response('stc')

            elif adv_model == 'qr':
                cum_def_actions = np.sum(all_def_strategy[:, :d+1], axis=1)
                adv_strategy = self.adversary_response('qr',
                                def_mixed_strategy=cum_def_actions)

            elif adv_model == 'suqr':
                cum_def_actions = np.sum(all_def_strategy[:, :d+1], axis=1)
                adv_strategy = self.adversary_response('suqr',
                                def_mixed_strategy=cum_def_actions)

            # defender play strategy
            attacks = self.strategy_to_attacks(adv_strategy)

            all_attacks[:, d]      = attacks
            all_adv_strategy[:, d] = adv_strategy
            all_est_reward[:, d]   = est_reward

            rewards[d] = self.get_defender_utility(attacks, def_strategy)

            if flag1:
                num_ol += 1
                reward_ol += rewards[d]
                arm_pulls.append('ol')
            else:
                num_ml += 1
                reward_ml += rewards[d]
                arm_pulls.append('ml')

            # update our belief of our reward
            K = self.geometric_resampling(gamma, eta, w, est_reward)
            K = np.sum(K, axis=1)

            # est_reward += np.sum(K * attacks * def_strategy, axis=1)
            est_reward += K * attacks * def_strategy


        if VERBOSE:
            print('num pulls ML = {}, OL = {}'.format(num_ml, num_ol))
            print('reward ML = {:.3f}, OL = {:.3f}'.format(reward_ml, reward_ol))
            print('estimated reward')
            print('  ', np.round(est_reward / est_reward.sum(), 2))
            print('true reward')
            print('  ', np.round(self.attack_prob, 2))

        regrets = self.compute_hindsight_regret(rewards, all_attacks)

        return {'rewards': rewards, 'regrets': regrets,
                'all_attacks': all_attacks,
                'arm_pulls': arm_pulls,
                'all_def_strategy': all_def_strategy,
                'all_est_reward': all_est_reward}


    def ml_predictions(self):
        """ Make ML predictions based on historical data
        """
        assert self.features is not None

        labels = self.historical_data.flatten()

        inputs = np.tile(self.features, (self.historical_data.shape[1], 1))

        classifier = DecisionTreeClassifier()
        classifier.fit(inputs, labels)

        predictions = classifier.predict_proba(self.features)
        predictions = predictions[:, 1]  # prediction of a positive label

        return predictions


    def check_params(self, adv_model):
        """ For parameters to be passed into adversary_response()
        """
        assert adv_model in {'stc', 'qr', 'suqr'}

        if adv_model == 'stc':
            assert self.attack_prob is not None
        elif adv_model in {'qr', 'suqr'}:
            # NOTE: lambda that experts suggest is 0.91 (Nguyen 2013)
            assert self.lamb is not None
            assert self.lamb >= 0

            # assert def_mixed_strategy is not None
            # assert def_mixed_strategy.shape == (self.num_loc,)

            if adv_model == 'suqr':
                assert self.w1 is not None and self.w2 is not None and self.w3 is not None


    def ml_exploit(self, adv_model, predictions):
        """ ML-exploit in AAMAS 2019 paper

        :return reward
        """
        self.check_params(adv_model)

        if VERBOSE:
            print()
            print('-------------------')
            print('ML learner')
            print('-------------------')

        rewards          = np.zeros(self.D)
        all_def_strategy = np.zeros((self.num_loc, self.D))
        all_attacks      = np.zeros((self.num_loc, self.D))
        all_adv_strategy = np.zeros((self.num_loc, self.D))

        for d in range(self.D):
            # cum_def_actions = np.sum(all_def_strategy[:, :d+1], axis=1)

            # adversary's view of our strategy is **all historical strategy**
            if d == 0:
                def_mixed_strategy = np.ones((self.num_loc, self.D))
                def_mixed_strategy = np.sum(def_mixed_strategy, axis=1)
            else:
                def_mixed_strategy = np.sum(all_def_strategy[:, :d+1], axis=1)
            def_mixed_strategy /= def_mixed_strategy.sum()

            # adversary pick targets
            if adv_model == 'stc':
                adv_strategy = self.adversary_response('stc')

                est_attack_prob = predictions

            elif adv_model == 'qr':

                adv_strategy = self.adversary_response('qr',
                                def_mixed_strategy=def_mixed_strategy)

                est_attack_prob = self.U_adv_u * (1 - def_mixed_strategy) + \
                                  self.U_adv_c * def_mixed_strategy

            elif adv_model == 'suqr':
                adv_strategy = self.adversary_response('suqr',
                                def_mixed_strategy=def_mixed_strategy)

                est_attack_prob = self.w1 * def_mixed_strategy + self.w2 * self.U_adv_u + \
                                  self.w3 * self.U_adv_c

            # normalize
            est_attack_prob /= est_attack_prob.sum()

            # compute defender strategy
            def_strategy = self.get_defender_strategy('ml',
                              start=None, end=None, start_t=None, end_t=None,
                              est_attack_prob=est_attack_prob,
                              alpha=None, est_reward=None, z=None)

            def_strategy = np.sum(def_strategy, axis=1)

            attacks = self.strategy_to_attacks(adv_strategy)

            all_def_strategy[:, d] = def_strategy
            all_adv_strategy[:, d] = adv_strategy
            all_attacks[:, d] = attacks

            rewards[d] = self.get_defender_utility(attacks, def_strategy)

        regrets = self.compute_hindsight_regret(rewards, all_attacks)

        return {'rewards': rewards, 'regrets': regrets,
                'attacks': all_attacks,
                'all_def_strategy': all_def_strategy}


    def pure_explore(self, adv_model, eta, w):
        """ Pure-explore in AAMAS 2019 paper

        :params same as online_learner(), except no gamma

        :return reward, adv_response
        """

        if VERBOSE:
            print()
            print('-------------------')
            print('pure explore')
            print('-------------------')

        gamma = 1  # always explore
        out = self.online_learner(adv_model, eta, w, gamma=gamma)

        return out


    def visualize_all_paths(self, predictions, all_def_strategy, title='', arm_pulls=None):
        if predictions.shape == (self.num_loc, self.D):
            change_predictions = True
        elif predictions.shape == (self.num_loc,):
            change_predictions = False
        else:
            raise Exception('Predictions must either be constant or have one per round.')

        num_rows = int(np.ceil(self.D / 10))
        num_cols = min(self.D, 10)

        fig, axs = plt.subplots(num_rows, num_cols, sharex='all', sharey='all',
                                figsize=(num_cols, num_rows))
        fig.suptitle('All paths - {}'.format(title))

        for d in range(self.D):
            row = d // num_cols
            col = d % num_cols

            # axs[row, col].plot(path_x, path_y, 'k-', linewidth=3, alpha=0.7)
            ax = axs[row, col]
            if arm_pulls is not None:
                ax.set_title('{} - {}'.format(d, arm_pulls[d]), {'fontsize': 8})
            else:
                ax.set_title(d, {'fontsize': 8})

            if change_predictions:
                self.draw_patrol(ax, predictions[:, d], all_def_strategy[:, d])
            else:
                self.draw_patrol(ax, predictions, all_def_strategy[:, d])

        fig.tight_layout()
        fig.subplots_adjust(top=0.85)
        plt.savefig('{}/paths_D{}_{}.png'.format(out_path, self.D, title))
        # plt.show()


    def draw_patrol(self, ax, predictions, def_strategy):
        pred_grid = predictions.reshape(self.width, self.height).T

        im = ax.imshow(pred_grid, cmap='Reds')
        # fig.colorbar(im, ax=ax, orientation='vertical')

        # draw gridlines
        ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
        ax.set_xticks(np.arange(-0.5, self.width, 1))
        ax.set_yticks(np.arange(-0.5, self.height, 1))

        # hide tick labels
        ax.get_xaxis().set_ticklabels([])
        ax.get_yaxis().set_ticklabels([])

        # plot patrol post(s)
        start_post = self.id_to_coord[self.start]
        end_post   = self.id_to_coord[self.end]
        ax.plot([start_post[0]], [start_post[1]], c='dodgerblue', marker='s', ms=4, zorder=1)
        ax.plot([end_post[0]], [end_post[1]], c='dodgerblue', marker='s', ms=4, zorder=2)

        # draw defender path

        ######################################
        # convert defender strategy to path
        ######################################
        # path_x = []
        # path_y = []
        # for t in range(def_strategy.shape[1]):
        #     # convert matrix def_strategy to path with locations
        #     loc_id = np.where(def_strategy[:, t])[0]
        #     assert loc_id.size == 1
        #     loc_coord = self.id_to_coord[loc_id.item(0)]
        #
        #     path_x.append(loc_coord[0])
        #     path_y.append(loc_coord[1])

        ######################################
        # compressed visit count
        ######################################
        path_x = []
        path_y = []
        size = []
        assert def_strategy.size == self.num_loc
        for i in range(self.num_loc):
            # only count if we visited this cell
            if def_strategy[i] > 0:
                loc_coord = self.id_to_coord[i]
                path_x.append(loc_coord[0])
                path_y.append(loc_coord[1])
                size.append(def_strategy[i])

        # ax.plot(path_x, path_y, 'ko', linewidth=3, alpha=0.2)
        # ax.plot(path_x, path_y, 'k-', linewidth=3, alpha=0.7)

        # cmap? linewidth?
        ax.scatter(path_x, path_y, size, c='k', alpha=0.5, zorder=3)


    def visualize_path(self, predictions, def_strategy):
        print(def_strategy)

        # make grid
        data = predictions.reshape(self.width, self.height)

        # draw predictions
        pred_grid = np.zeros((self.width, self.height))
        for i in range(len(predictions)):
            coords = self.id_to_coord[i]
            pred_grid[coords[0], coords[1]] = predictions[i]

        fig, ax = plt.subplots()
        im = ax.imshow(pred_grid, cmap='Reds')
        fig.colorbar(im, ax=ax, orientation='vertical')

        # draw gridlines
        ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
        ax.set_xticks(np.arange(-0.5, self.width, 1))
        ax.set_yticks(np.arange(-0.5, self.height, 1))

        print(self.width, self.height)

        # plot patrol post(s)
        start_post = self.id_to_coord[self.start]
        end_post   = self.id_to_coord[self.end]
        ax.scatter([start_post[0]], [start_post[1]], c='b', marker='o')
        ax.scatter([end_post[0]], [end_post[1]], c='b', marker='o')

        # draw defender path
        # convert defender strategy to path
        path_x = []
        path_y = []
        for t in range(def_strategy.shape[1]):
            # convert matrix def_strategy to path with locations
            loc_id = np.where(def_strategy[:, t])[0]
            assert loc_id.size == 1
            loc_coord = self.id_to_coord[loc_id.item(0)]

            path_x.append(loc_coord[0])
            path_y.append(loc_coord[1])

        # ax.plot(path_x, path_y, 'ko', linewidth=3, alpha=0.2)
        ax.plot(path_x, path_y, 'k-', linewidth=3, alpha=0.7)

        plt.show()


#############################################
# run script
#############################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', '-W', help='park width', type=int, default=5)
    parser.add_argument('--height', '-H', help='test height', type=int, default=5)
    parser.add_argument('--horizon', '-L', help='patrol length (horizon)', type=int, default=10)
    parser.add_argument('--num_attackers', '-M', help='number of attackers', type=int, default=11)
    parser.add_argument('--num_rounds', '-D', help='number of rounds', type=int, default=20)
    parser.add_argument('--lamb', '-l', help='lambda', type=float, default=0.7)
    parser.add_argument('--norepeat', help='no repeat (for error bars)', action='store_true', default=False)

    args = parser.parse_args()


    # park params
    width       = args.width
    height      = args.height
    patrol_post = 12

    # game params
    horizon = args.horizon
    M       = args.num_attackers
    D       = args.num_rounds

    # MINION params
    eta   = 3.1
    gamma = 0.5
    w     = 10
    beta  = 4.2
    e     = 5

    # QR and SUQR model params
    # w1 = -9.83, w2 = .37, w3 = .15
    lamb = args.lamb
    w1 = -1.
    w2 = 2.
    w3 = 1.

    norepeat = args.norepeat


    #############################################
    # set up minion
    #############################################

    num_targets   = width * height
    num_features  = 3
    num_hids      = 100
    num_layers    = 10
    num_instances = 1
    num_samples   = 6
    attacker_w    = -4.0
    data = generate_synthetic_data(num_targets, num_features, num_hids,
                           num_layers, num_instances, num_samples,
                           attacker_w=attacker_w)

    features      = data[0].squeeze()
    defender_vals = data[1]
    attacker_vals = data[2]
    attacker_w    = data[3]

    target_vals = defender_vals
    target_vals = np.abs(target_vals)
    print('target_vals')
    print('  ', np.round(target_vals, 2))

    # stochastic (stationary) adversary
    attack_prob = target_vals / target_vals.sum()
    print('real attack prob')
    print('  ', np.round(attack_prob, 2))
    print('  sum {}'.format(attack_prob.sum()))

    minion = Minion(width=width, height=height, D=D, horizon=horizon, M=M,
                    patrol_post=patrol_post, attack_prob=attack_prob,
                    lamb=lamb, w1=w1, w2=w2, w3=w3)

    minion.features = features

    minion.U_adv_c = -target_vals
    minion.U_adv_u = target_vals
    minion.U_def_c = target_vals
    minion.U_def_u = -target_vals

    historical  = minion.set_historical_data(num_timesteps=100)

    #predictions = minion.ml_predictions()
    predictions = attack_prob
    ml_mae = np.abs(predictions - attack_prob).mean()

    print('predictions, MAE = {:.3f}'.format(ml_mae))
    print('  ', np.around(predictions, 2))


    #############################################
    # visualize paths
    #############################################

    # def_strategy = self.get_defender_strategy(method='ol',
    #         alpha=.5, est_reward=est_reward, z=z)
    # def_strategy = minion.get_defender_strategy(method='ol',
    #         alpha=.5, est_reward=minion.U_def_c, z=z)
    # def_strategy = minion.get_defender_strategy(method='ml',
    #         est_attack_prob=attack_prob)


    # for adv_model in ['stc', 'qr', 'suqr']:
    #     print('------------------------')
    #     print('adversary model: {}'.format(adv_model))
    #     for method in ['hybrid', 'online', 'ml', 'explore']:
    #         print('  method: {}'.format(method))
    #
    #         if method == 'ml':
    #             out = minion.ml_exploit(adv_model, predictions)
    #             minion.visualize_all_paths(predictions, out['all_def_strategy'],
    #                     title='ML exploit - {}'.format(adv_model))
    #
    #         elif method == 'online':
    #             out = minion.online_learner(adv_model, eta, w, gamma)
    #             minion.visualize_all_paths(out['all_est_reward'], out['all_def_strategy'],
    #                     title='Online (MINION-sm) - {}'.format(adv_model))
    #
    #         elif method == 'hybrid':
    #             out = minion.hybrid(adv_model, eta, beta, w, gamma, e)
    #             minion.visualize_all_paths(out['all_est_reward'], out['all_def_strategy'],
    #                     title='Hybrid (MINION) - {}'.format(adv_model),
    #                     arm_pulls=out['arm_pulls'])
    #
    #         elif method == 'explore':
    #             out = minion.pure_explore(adv_model, eta, w)
    #             minion.visualize_all_paths(out['all_est_reward'], out['all_def_strategy'],
    #                     title='Pure explore - {}'.format(adv_model))
    #
    # sys.exit(0)

    #############################################
    # run experiments
    #############################################

    hybrid_results  = {'stc': [], 'qr': [], 'suqr': []}     # MINION
    online_results  = {'stc': [], 'qr': [], 'suqr': []}     # MINION-sm
    ml_results      = {'stc': [], 'qr': [], 'suqr': []}     # ml-exploit
    explore_results = {'stc': [], 'qr': [], 'suqr': []}     # pure explore

    results = {'hybrid': hybrid_results, 'online': online_results,
               'ml': ml_results, 'explore': explore_results}

    # execute multiple times for error bars
    if norepeat:
        num_repeats = 1
    else:
        num_repeats = 10

    for i in range(num_repeats):
        print('\n========================================')
        print('iteration {} / {}'.format(i, num_repeats))

        for adv_model in ['stc', 'qr', 'suqr']:
            print()
            print('------------------------')
            print('adversary model: {}'.format(adv_model))
            print('------------------------')

            for method in ['hybrid', 'online', 'ml', 'explore']:
                if method == 'hybrid':
                    print('MINION (hybrid)')
                    out = minion.hybrid(adv_model, eta, beta, w, gamma, e)
                elif method == 'online':
                    print('MINION-sm (online)')
                    out = minion.online_learner(adv_model, eta, w, gamma)
                elif method == 'ml':
                    print('ML exploit, MAE = {:.4f}'.format(ml_mae))
                    out = minion.ml_exploit(adv_model, predictions)
                elif method == 'explore':
                    print('pure explore')
                    out = minion.pure_explore(adv_model, eta, w)

                results[method][adv_model].append(out)
                print('  {}'.format(np.round(out['rewards'], 2)))
                print('  total reward: {:.2f}'.format(out['rewards'].sum()))
                print('  total regret: {:.2f}'.format(out['regrets'].sum()))


    #############################################
    # visualize regret plots
    #############################################

    def make_plot(title, adversary):
        rounds = np.arange(D)

        def get_mean_and_std(iters):
            regret = np.array([iter['regrets'] for iter in iters])
            mean   = np.mean(regret, axis=0)
            std    = np.std(regret, axis=0)
            return mean, std

        line  = {'explore': 'b--', 'ml': 'g-.', 'hybrid': 'k-', 'online': 'r:'}
        label = {'explore': 'pure-explore', 'ml': 'ML-exploit',
                 'hybrid': 'MINION (hybrid)', 'online': 'MINION-sm (online)'}

        # display plot
        plt.figure()
        fig, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [7, 1]})

        # plot each line with errorbar
        for model in ['explore', 'ml', 'hybrid', 'online']:
            mean, std = get_mean_and_std(results[model][adversary])
            a0.errorbar(rounds, mean, yerr=std, fmt=line[model],
                        elinewidth=0.5, label=label[model])

        a0.legend()
        a0.set_xlabel('round (D)')
        a0.set_ylabel('regret')

        a1.plot(rounds, hybrid_results[adversary][0]['arm_pulls'], 'o', alpha=.5)
        a1.set_xlabel('selected arm (first iter)')

        fig.suptitle('{} m={}, L={}, T={}, MAE={:.3f}, lambda={}'.format(
                title, M, width*height, horizon, ml_mae, lamb))
        fig.tight_layout()
        fig.subplots_adjust(top=0.93)
        if norepeat:
            fig.savefig('{}/norepeat_m{}_L{}_T{}_MAE{:.3f}_lamb{}_D{}-_{}.png'.format(
                out_path, M, width*height, horizon, ml_mae, lamb, D, adversary))
        else:
            fig.savefig('{}/m{}_L{}_T{}_MAE{:.3f}_lamb{}_D{}-_{}.png'.format(
                out_path, M, width*height, horizon, ml_mae, lamb, D, adversary))

    make_plot('stochastic adversary', 'stc')
    make_plot('QR adversary', 'qr')
    make_plot('SUQR adversary', 'suqr')


if __name__ == '__main__':
    if SHOW_RUNTIME:
        import cProfile
        cProfile.run('main()', sort='cumtime')
    else:
        main()
