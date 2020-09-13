""" Lily Xu

implement LIZARD algorithm, implement baselines
generate performance graphs and results
"""

import sys, os
import argparse
import pickle
from multiprocessing import Pool
from datetime import datetime

import scipy
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from scipy.stats import bernoulli
import matplotlib.pyplot as plt

from discrete_lipschitz import DiscreteLipschitz
from decomposed_zooming import DecomposedLipschitz
from generate_graph import generate_synthetic_data
from adversary import EffortAdversary, RealEffortAdversary

np.random.seed(42)

out_path = './output'

if not os.path.exists(out_path):
    os.makedirs(out_path)

def plot_compare():
    all_avg, all_sem = run_processes()
    results = {'all_avg': all_avg, 'all_sem': all_sem}
    timestamp = datetime.now()
    data_type = 'synthetic' if synthetic else 'real-world'
    pickle.dump(results, open('{}/run_n{}_b{}_{}_H{}_repeat{}_{}.p'.format(out_path, N, B, data_type, historical_T, num_repeats, timestamp), 'wb'))

    x_vals = np.arange(T)

    # plot reward
    fig = plt.figure()

    for model in models:
        reward_avg = smooth(all_avg[model]['reward'] / N)
        reward_sem = smooth(all_sem[model]['reward'] / N)

        color = colors[model]

        plt.plot(x_vals, reward_avg, label=model, c=color)
        plt.fill_between(x_vals, reward_avg-reward_sem, reward_avg+reward_sem, facecolor=color, alpha=0.15)

        if model in ['decomposed']:
            exploit_avg = smooth(all_avg[model+'_exploit']['reward'] / N)
            exploit_sem = smooth(all_sem[model+'_exploit']['reward'] / N)
            plt.plot(x_vals, exploit_avg, label=model+' exploit', c=colors[model+'_exploit'])

    plt.title('reward - {} - N={} B={}, H={}, {} repeats'.format(data_type, N, B, historical_T, num_repeats))
    plt.xlabel('timestep')
    plt.ylabel('smoothed avg reward')
    plt.legend()
    fig.tight_layout()
    plt.savefig('{}/reward_{}_n{}_b{}_H{}_{}.png'.format(out_path, data_type, N, B, historical_T, timestamp))
    # plt.show()

    # plot regret
    fig = plt.figure()

    for model in models:
        regret_avg = smooth(all_avg[model]['regret'] / N)
        regret_sem = smooth(all_sem[model]['regret'] / N)

        color = colors[model]

        plt.plot(x_vals, regret_avg, label=model, c=color)
        plt.fill_between(x_vals, regret_avg-regret_sem, regret_avg+regret_sem, facecolor=color, alpha=0.15)

        #if model in ['decomposed', 'decomposed_history', 'decomposed_features', 'random']:
        if model in ['decomposed']:
            exploit_avg = smooth(all_avg[model+'_exploit']['regret'] / N)
            exploit_sem = smooth(all_sem[model+'_exploit']['regret'] / N)
            plt.plot(x_vals, exploit_avg, label=model+' exploit', c=colors[model+'_exploit'])

    plt.title('regret - {} - N={} B={}, {} repeats'.format(data_type, N, B, num_repeats))
    plt.xlabel('timestep')
    plt.ylabel('smoothed avg regret')
    plt.legend()
    fig.tight_layout()
    plt.savefig('{}/regret_{}_n{}_b{}_{}.png'.format(out_path, data_type, N, B, timestamp))
    plt.show()

    # plot regret
    plt.figure()
    plt.plot(x_vals, optimal_reward - lipschitz_reward, label='lipschitz', c='o')
    plt.plot(x_vals, optimal_reward - random_reward, label='random', c='g')
    plt.title('regret')
    plt.xlabel('timestep')
    plt.ylabel('cumulative regret')
    plt.legend()
    plt.savefig('stochastic_adv_regret.png')
    plt.show()


def smooth(rewards, weight=0.7):
    """ smoothed exponential moving average """
    prev = rewards[0]
    smoothed = np.zeros(len(rewards))
    for i, val in enumerate(rewards):
        smoothed_val = prev * weight + (1 - weight) * val
        smoothed[i] = smoothed_val
        prev = smoothed_val

    return smoothed


class Run:
    def __init__(self, N, T, synthetic, seed=42, VERBOSE=False):
        """ synthetic: generates synthetic if True
        otherwise, use real-world data """
        np.random.seed(seed)

        self.VERBOSE = VERBOSE
        self.synthetic = synthetic

        self.N = N
        self.T = T

        data = generate_synthetic_data(N, num_features, num_hids,
                               num_layers, num_instances, num_samples,
                               attacker_w=-4.0)
        features      = data[0].squeeze()
        defender_vals = data[1]
        attacker_vals = data[2]
        attacker_w    = data[3]

        # scale to [0, 1]
        attacker_vals -= attacker_vals.min()
        attacker_vals /= attacker_vals.max()

        # scale to [1, 11]
        attacker_vals *= 10
        attacker_vals += 1

        self.features = features
        self.attacker_vals = attacker_vals

        if self.synthetic:
            if self.VERBOSE:
                print('features', features)
                print('---')
                print('defender vals', defender_vals)
                print('---')
                print('attacker vals', attacker_vals)
                print('---')
                print('attacker w', attacker_w)

            self.adversary = EffortAdversary(self.attacker_vals)
        else:
            region = np.random.randint(40)  # we have 40 instances in each pickle
            self.adversary = RealEffortAdversary(self.N, region)
            self.region = region


        self.optimal = None  # record the optimal beta
        self.historical_arm = None  # arm selected with historical exploit

        self.budget = B
        self.num_discretization = 10

        self.history = None  # historical data


    def get_historical(self, historical_T):
        """ generate historical data """
        print('------------------------------')
        print('generating {} timesteps of historical data...'.format(historical_T))

        def get_historical_effort():
            """ generate historical effort """
            effort = np.zeros((historical_T, self.N))

            partial_distribution = np.random.uniform(size=self.N//2)
            # partial_distribution = np.random.exponential(size=self.N//2)
            partial_distribution /= partial_distribution.sum()

            distribution = np.zeros(self.N)
            idx = np.random.choice(self.N, self.N//2, replace=False)  # where to put our effort
            distribution[idx] = partial_distribution

            effort_step = 1. / self.num_discretization
            for t in range(historical_T):
                for t2 in range(self.num_discretization * self.budget):
                    # ensure we don't put more than 1 unit of effort anywhere
                    potential_targets = np.where(effort[t, :] + effort_step <= 1)[0]
                    target_prob = distribution[potential_targets] / distribution[potential_targets].sum()

                    target = np.random.choice(potential_targets, p=target_prob)
                    effort[t, target] += effort_step

            # eliminating floating point glitches
            effort = np.round(effort, 3)

            return effort

        B           = [[] for _ in range(self.N)]
        cum_rewards = [{} for _ in range(self.N)]
        n           = [{} for _ in range(self.N)]

        #### put arms in order
        # make random arms in a grid to initialize
        for u in range(self.num_discretization+1):
            eff_levels = np.linspace(0, 1, self.num_discretization+1)
            # eliminating floating point glitches
            eff_levels = np.round(eff_levels, 3)
            eff = eff_levels[u]

            for i in range(self.N):
                B[i].append(eff)
                n[i][eff] = 0
                cum_rewards[i][eff] = 0

        if not self.synthetic:
            from get_real_effort import get_real_effort

            n = get_real_effort(self.N, self.region)

            for i in range(self.N):
                for eff in B[i]:
                    for _ in range(n[i][eff]):
                        reward_prob = self.adversary.visit_target(i, eff)

                        # update our estimate
                        observed_reward = np.random.choice([0, 1], p=[1-reward_prob, reward_prob])
                        # n[i][effort[t, i]] += 1
                        cum_rewards[i][eff] += observed_reward


        else:
            effort = get_historical_effort()

            # ensure no effort greater than 1
            assert len(np.where(effort > 1.)[0]) == 0

            # sample with historical effort
            for t in range(historical_T):
                for i in range(self.N):
                    reward_prob = self.adversary.visit_target(i, effort[t, i])

                    # update our estimate
                    observed_reward = np.random.choice([0, 1], p=[1-reward_prob, reward_prob])
                    n[i][effort[t, i]] += 1
                    cum_rewards[i][effort[t, i]] += observed_reward

        self.history = {'B': B, 'cum_rewards': cum_rewards, 'n': n}

        return self.history


    def decomposed_lipschitz(self, increasingness=False, history=False, use_features=False, VERBOSE=False):
        if history:
            lipschitz = DecomposedLipschitz(
                self.features, self.adversary, T=self.T, optimal=self.optimal, budget=self.budget,
                increasingness=increasingness, history=self.history, use_features=use_features,
                VERBOSE=VERBOSE)
        else:
            lipschitz = DecomposedLipschitz(
                self.features, self.adversary, T=self.T, optimal=self.optimal, budget=self.budget,
                increasingness=increasingness, use_features=use_features,
                VERBOSE=VERBOSE)

        all_reward = lipschitz.zooming()
        t_uncover  = lipschitz.t_uncover
        t_ucb      = lipschitz.t_ucb
        exploit_rewards = lipschitz.exploit_rewards
        return {'reward': all_reward, 't_uncover': t_uncover, 't_ucb': t_ucb,
                'exploit_reward': exploit_rewards}


    def solve_exploit(self, B, n, cum_rewards):
        """ given historical data, solve for optimal arm pull

        used in to exploit in random baseline, epsilon-greedy """

        model = gp.Model('exploit')

        # silence output
        model.setParam('OutputFlag', 0)

        x = [[model.addVar(vtype=GRB.BINARY, name='x_{}_{}'.format(i, j))
                for j in range(len(B[i]))] for i in range(self.N)]

        model.setObjective(gp.quicksum([x[i][j] * cum_rewards[i][eff] / max(1, n[i][eff])
                    for i in range(self.N) for j, eff in enumerate(B[i])]),
                    GRB.MAXIMIZE)

        model.addConstrs((gp.quicksum(x[i][j] for j, eff in enumerate(B[i])) == 1
                    for i in range(self.N)), 'one_per_target') # pull one arm per target

        model.addConstr(gp.quicksum([x[i][j] * B[i][j]
                    for i in range(self.N) for j, eff in enumerate(B[i])]) <= self.budget, 'budget')  # stay in budget

        model.optimize()

        if model.status != GRB.OPTIMAL:
            raise Exception('Uh oh! Model status is {}'.format(model.status))

        # convert x to beta
        exploit_arm = np.full(self.N, np.nan)
        for i in range(self.N):
            for j in range(len(B[i])):
                if abs(x[i][j].x - 1) < 1e-2:
                    exploit_arm[i] = B[i][j]

            assert not np.isnan(exploit_arm[i]), 'MIP x[{}] vals are {}'.format(i, [x[i][j].x for j in range(len(B[i]))])

        return exploit_arm


    def historical_exploit(self):
        assert self.history is not None

        B = self.history['B']
        n = self.history['n']
        cum_rewards = self.history['cum_rewards']

        arm = self.solve_exploit(self.history['B'], self.history['n'], self.history['cum_rewards'])

        arm_rewards = np.zeros(self.N)
        for i in range(self.N):
            arm_rewards[i] = self.adversary.visit_target(i, arm[i])

        rewards = np.ones(self.T) * np.sum(arm_rewards)

        self.historical_arm = arm

        return {'reward': rewards}


    def random(self, exploit=True):
        # select effort to allot per target
        selected = np.random.uniform(size=(self.N, self.T))
        selected = selected / selected.sum(axis=0)   # normalize

        if exploit:
            B           = [[] for _ in range(self.N)]  # list of arms (decomposed)
            cum_rewards = [{} for _ in range(self.N)]  # cumulative reward
            n           = [{} for _ in range(self.N)]  # number of pulls

            exploit_reward = np.zeros(self.T)

        all_reward = np.zeros(self.T)

        for t in range(self.T):
            for i in range(self.N):
                eff = selected[i, t]

                reward_prob = self.adversary.visit_target(i, eff)
                all_reward[t] += reward_prob

                if exploit:
                    if eff not in n[i]:
                        B[i].append(eff)
                        n[i][eff] = 0
                        cum_rewards[i][eff] = 0

                    observed_reward = np.random.choice([0, 1], p=[1-reward_prob, reward_prob])
                    n[i][eff] += 1
                    cum_rewards[i][eff] += observed_reward

            if exploit:
                exploit_arm = self.solve_exploit(B, n, cum_rewards)

                for i in range(self.N):
                    exploit_reward[t] += self.adversary.visit_target(i, exploit_arm[i])

        if exploit:
            return {'reward': all_reward, 'exploit_reward': exploit_reward}
        else:
            return {'reward': all_reward}


    def CUCB(self):
        """ implement CUCB algorithm (Chen et al. 2016)
        no history, no features, no lipschitzness
        implement oracle as MIP to maximize UCB """
        B           = [[] for _ in range(self.N)]  # list of arms (decomposed)
        cum_rewards = [{} for _ in range(self.N)]  # cumulative reward
        n           = [{} for _ in range(self.N)]  # number of pulls

        UCB         = [{} for _ in range(self.N)]

        all_reward = np.zeros(self.T)

        # initialize UCB values
        # make random arms in a grid to initialize
        for u in range(self.num_discretization+1):
            eff_levels = np.linspace(0, 1, self.num_discretization+1)
            # eliminating floating point glitches
            eff_levels = np.round(eff_levels, 3)
            eff = eff_levels[u]

            for i in range(self.N):
                B[i].append(eff)
                n[i][eff] = 0
                cum_rewards[i][eff] = 0

                UCB[i][eff] = 1.

        for t in range(self.T):
            model = gp.Model('CUCB')
            model.setParam('OutputFlag', 0)

            # if n == 0, mu = 1.
            # x: indicator saying pull arm j at target i
            x = [[model.addVar(vtype=GRB.BINARY, name='x_{}_{}'.format(i, j))
                    for j in range(len(B[i]))] for i in range(self.N)]

            model.setObjective(gp.quicksum([x[i][j] * UCB[i][eff]
                for i in range(self.N) for j, eff in enumerate(B[i])]), GRB.MAXIMIZE)

            model.addConstrs((gp.quicksum(x[i][j] for j in range(len(B[i]))) == 1
                                for i in range(self.N)), 'one_per_target') # pull one arm per target

            model.addConstr(gp.quicksum([x[i][j] * B[i][j]
                                for i in range(self.N) for j in range(len(B[i]))]) <= self.budget, 'budget')  # stay in budget

            model.optimize()

            if model.status != GRB.OPTIMAL:
                raise Exception('Uh oh! Model status is {}'.format(model.status))

            beta = np.full(self.N, np.nan)

            # convert x to beta
            for i in range(self.N):
                for j, eff in enumerate(B[i]):
                    if abs(x[i][j].x - 1) < 1e-2:
                        beta[i] = B[i][j]

                assert not np.isnan(beta[i]), 'MIP x[{}] vals are {}'.format(i, [x[i][j].x for j in range(len(B[i]))])


            all_reward[t] = self.update_reward(beta, B, n, cum_rewards)

            # # assert within budget
            # print('{} budget {}, sum {:.1f}, rew {:.2f}, beta {}'.format(t, self.budget, np.sum(beta), all_reward[t], np.round(beta, 2)))

            # update UCB values
            for i in range(self.N):
                eff = beta[i]
                mu = cum_rewards[i][eff] / max(1, n[i][eff])

                eps = .1
                r = np.sqrt(-np.log(eps) / (2. * max(1, n[i][eff])))
                UCB[i][eff] = mu + r

        return {'reward': all_reward}



    def epsilon_greedy(self, eps=0.1):
        """ implement epsilon-greedy strategy """

        all_reward = np.zeros(self.T)

        B           = [[] for _ in range(self.N)]  # list of arms (decomposed)
        cum_rewards = [{} for _ in range(self.N)]  # cumulative reward
        n           = [{} for _ in range(self.N)]  # number of pulls

        for t in range(self.T):
            exploit = bernoulli.rvs(eps)

            if exploit and t > 0:
                beta = self.solve_exploit(B, n, cum_rewards)

            else:
                beta = np.random.uniform(size=self.N)
                beta /= np.sum(beta)  # normalize
                beta *= self.budget

                # ensure we never exceed effort = 1 on any target
                while len(np.where(beta > max_effort)[0]) > 0:
                    excess_idx = np.where(beta > 1)[0][0]
                    excess = beta[excess_idx] - max_effort

                    beta[excess_idx] = max_effort

                    # add "excess" amount of effort randomly on other targets
                    add = np.random.uniform(size=self.N - 1)
                    add = (add / np.sum(add)) * excess

                    beta[:excess_idx] += add[:excess_idx]
                    beta[excess_idx+1:] += add[excess_idx:]

            all_reward[t] = self.update_reward(beta, B, n, cum_rewards)

        return {'reward': all_reward}


    def update_reward(self, beta, B, n, cum_rewards):
        assert np.sum(beta) <= self.budget + 1e-6
        # elimnate floating point issues
        beta = np.round(beta, 3)

        # observe reward
        reward = 0
        for i in range(self.N):
            reward_prob = self.adversary.visit_target(i, beta[i])
            reward += reward_prob

            if beta[i] not in n[i]:
                B[i].append(beta[i])
                n[i][beta[i]] = 0
                cum_rewards[i][beta[i]] = 0

            # update our estimate
            observed_reward = np.random.choice([0, 1], p=[1-reward_prob, reward_prob])
            n[i][beta[i]] += 1
            cum_rewards[i][beta[i]] += observed_reward

        return reward


    def standard_lipschitz(self):
        """ standard Lipschitz: regular zooming algorithm
        metric space: effort space
        """

        # NOTE: regular lipschitz can't discretize over all possible meta-arms
        # we'd have (num_discrete)^N arms, e.g. 10^25

        # instead take 1000 randomly generated arms

        # pick random 'arms' corresponding to coverage
        num_random = 1000
        random_beta = np.random.uniform(0, 1, size=(num_random, self.N))

        # normalize to sum to budget
        random_beta = random_beta.T / random_beta.sum(axis=1) * self.budget
        random_beta = random_beta.T

        max_effort = 1.

        # ensure we never exceed effort = 1 on any target
        for i in range(num_random):
            while len(np.where(random_beta[i,:] > max_effort)[0]) > 0:
                excess_idx = np.where(random_beta[i,:] > 1)[0][0]
                excess = random_beta[i, excess_idx] - max_effort

                random_beta[i, excess_idx] = max_effort

                # add "excess" amount of effort randomly on other targets
                add = np.random.uniform(size=self.N - 1)
                add = (add / np.sum(add)) * excess

                random_beta[i, :excess_idx] += add[:excess_idx]
                random_beta[i, excess_idx+1:] += add[excess_idx:]

        lipschitz = DiscreteLipschitz(random_beta, self.adversary, T=self.T, alpha=self.attacker_vals, VERBOSE=self.VERBOSE)
        all_reward = lipschitz.zooming()

        return {'reward': all_reward}


    def FTPL_helper(self, gamma, eta, t, B, n, cum_rewards):
        """ subroutine of FTPL called at each timestep """
        num_discretization = 10
        effort_step = self.budget / num_discretization

        # sample flag
        flag = bernoulli.rvs(1 - gamma)

        # for each arm generate random noise from an exponential distribution
        z = np.random.exponential(eta, (self.N, num_discretization+1))

        if flag == 0 or t == 0:
            # exploration

            beta = np.zeros(self.T)

            # for each amount of effort we can spend...
            # (discrete world)
            for _ in range(num_discretization):
                # ensure we don't exceed 1
                potential_targets = np.where(beta + effort_step <= 1)[0]
                # select random target
                target = np.random.choice(potential_targets)
                beta[target] += effort_step
        else:
            # exploitation
            perturb_cum_rewards = [{} for _ in range(self.N)]

            # perturb cum_rewards by z
            for i in range(self.N):
                for j, eff in enumerate(B[i]):
                    perturb_cum_rewards[i][eff] = cum_rewards[i][eff] + (n[i][eff] * z[i][j])

            beta = self.solve_exploit(B, n, perturb_cum_rewards)

        return beta


    def FTPL(self, gamma, eta):
        """ online algorithm (MINION-sm)
        follow the perturbed leader """

        B           = [[] for _ in range(self.N)]  # list of arms (decomposed)
        cum_rewards = [{} for _ in range(self.N)]  # cumulative reward
        n           = [{} for _ in range(self.N)]  # number of pulls

        all_reward = np.zeros(self.N)

        for t in range(self.T):
            beta = self.FTPL_helper(gamma, eta, t, B, n, cum_rewards)
            all_reward[t] = self.update_reward(beta, B, n, cum_rewards)

        return {'reward': all_reward}


    def minion(self, gamma, eta, beta_ftpl):
        """ MINION algorithm from AAMAS 2019 paper (Gholami et al. 2019)
        hybrid model between online algorithm (FTPL) and ML exploit
        """
        assert self.historical_arm is not None

        reward_ml = 0.
        reward_ol = 0.
        n_ml = 0
        n_ol = 0

        num_initial = 15

        B           = [[] for _ in range(self.N)]  # list of arms (decomposed)
        cum_rewards = [{} for _ in range(self.N)]  # cumulative reward
        n           = [{} for _ in range(self.N)]  # number of pulls

        # initialize
        for u in range(self.num_discretization+1):
            eff_levels = np.linspace(0, 1, self.num_discretization+1)
            # eliminating floating point glitches
            eff_levels = np.round(eff_levels, 3)
            eff = eff_levels[u]

            for i in range(self.N):
                B[i].append(eff)
                n[i][eff] = 0
                cum_rewards[i][eff] = 0

        arm_pulls = []

        all_reward = np.zeros(self.T)

        for t in range(self.T):
            # ensure that ML and OL each get pulled at least once
            # to avoid division by zero
            if t == 0:
                flag1 = True
            elif t == 1:
                flag1 = False
            # explore uniformly between ML and OL to initialize rewards
            elif t < num_initial:
                flag1 = np.random.choice([True, False])
            else:
                c1 = np.random.exponential(beta_ftpl)
                c2 = np.random.exponential(beta_ftpl)
                flag1 = (reward_ml / n_ml + c1) < (reward_ol / n_ol + c2)

            if flag1:
                # use online model
                beta = self.FTPL_helper(gamma, eta, t, B, n, cum_rewards)

            else:
                # use ML model
                beta = self.historical_arm

            # update our belief of our reward
            all_reward[t] = self.update_reward(beta, B, n, cum_rewards)

            if flag1:
                n_ol += 1
                reward_ol += all_reward[t]
                arm_pulls.append('ol')
            else:
                n_ml += 1
                reward_ml += all_reward[t]
                arm_pulls.append('ml')

        return {'reward': all_reward}


    def optimal_effort(self):
        print('------------------------------')
        print('computing optimal arm...')

        all_start_x, all_end_x, all_start_y, all_end_y, all_slope = self.adversary.get_start_end()

        # start with only one segment per PWL function
        num_segs = np.ones(self.N, dtype=np.int8)

        M = 1e6
        prev_solution = None

        # assuming that each PWL function has same number of segments
        while len(np.where(num_segs < len(all_start_x[0]))):
            ##### CONSTRAINT GENERATION
            model = gp.Model('optimal_milp')

            # silence output
            if not self.VERBOSE:
                model.setParam('OutputFlag', 0)

            # model.setParam('IterationLimit', 1000)

            # are we greater than start of segment i,j?
            I_start = [[model.addVar(vtype=GRB.BINARY, name='I_start_{}_{}'.format(i, j))
                    for j in range(min(num_segs[i]+1, len(all_start_x[i])))] for i in range(self.N)]
            # are we less than the end of segment i,j?
            I_end = [[model.addVar(vtype=GRB.BINARY, name='I_end_{}_{}'.format(i, j))
                    for j in range(min(num_segs[i]+1, len(all_start_x[i])))] for i in range(self.N)]
            # are we within segment i,j? (both I_start and I_end)
            I_seg = [[model.addVar(vtype=GRB.BINARY, name='I_seg_{}_{}'.format(i, j))
            # I_seg = [[model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name='I_seg_{}_{}'.format(i, j))
                    for j in range(min(num_segs[i]+1, len(all_start_x[i])))] for i in range(self.N)]

            start_x = []
            end_x   = []
            start_y = []
            end_y   = []
            slope   = []
            for i in range(self.N):
                start_x.append(list(all_start_x[i][:num_segs[i]]))
                end_x.append(list(all_end_x[i][:num_segs[i]]))
                start_y.append(list(all_start_y[i][:num_segs[i]]))
                end_y.append(list(all_end_y[i][:num_segs[i]]))
                slope.append(list(all_slope[i][:num_segs[i]]))

                if num_segs[i] < len(all_start_x[i]):
                    start_x[i].append(end_x[i][-1])
                    end_x[i].append(all_end_x[i][-1])
                    start_y[i].append(end_y[i][-1])
                    end_y[i].append(all_end_y[i][-1])
                    slope[i].append((end_y[i][-1] - start_y[i][-1]) /
                                    (end_x[i][-1] - start_x[i][-1]))

                # print('i ------ ', i)
                # print(start_x[i])
                # print(end_x[i])

            # optimal value for x[i]
            x     = [model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name='x_{}'.format(i))
                    for i in range(self.N)]

            # x[i] and I_seg[i,j]
            w     = [[model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name='w_{}_{}'.format(i, j))
                    for j in range(len(start_x[i]))] for i in range(self.N)]

            # objective: I_seg * (start_y + (x - start_x) * slope)
            model.setObjective(gp.quicksum([I_seg[i][j] * start_y[i][j] +
                                    w[i][j] * slope[i][j] -
                                    I_seg[i][j] * start_x[i][j] * slope[i][j]
                for i in range(self.N) for j in range(len(start_x[i]))]),
                GRB.MAXIMIZE)

            padding = 1e-5

            model.addConstr(gp.quicksum(x) <= self.budget, 'budget')

            model.addConstrs((x[i] - (start_x[i][j] + padding) + M * (1 - I_start[i][j]) >= 0.
                    for i in range(self.N) for j in range(len(start_x[i]))), 'ub_I_start')

            model.addConstrs((end_x[i][j] - padding - x[i] + M * (1 - I_end[i][j]) >= 0.
                    for i in range(self.N) for j in range(len(start_x[i]))), 'ub_I_end')

            model.addConstrs((I_start[i][j] + I_end[i][j] - 1 >= I_seg[i][j]
                    for i in range(self.N) for j in range(len(start_x[i]))), 'ub_I_seg')

            model.addConstrs((w[i][j] <= x[i]
                    for i in range(self.N) for j in range(len(start_x[i]))), 'ub_w')

            model.addConstrs((w[i][j] <= I_seg[i][j]
                    for i in range(self.N) for j in range(len(start_x[i]))), 'ub_w2')

            model.optimize()

            if model.status != GRB.OPTIMAL:
                raise Exception('Uh oh! Model status is {}'.format(model.status))

            # did we ever pick the last segment?
            picked = False
            for i in range(self.N):
                if num_segs[i] == len(all_start_x[i]): continue
                if I_start[i][-1].x:  # if selected the last segment
                    num_segs[i] += 1
                    picked = True
            if not picked:
                break


        arm = np.zeros(self.N)
        for i in range(self.N):
            arm[i] = x[i].x

        # for v in model.getVars():
        #     print('%s %g' % (v.varName, v.x))

        self.optimal = arm

        # pull arm
        arm_rewards = np.zeros(self.N)
        for i in range(self.N):
            arm_rewards[i] = self.adversary.visit_target(i, arm[i])

        opt_reward = np.sum(arm_rewards)
        rewards = np.ones(self.T) * opt_reward

        print('optimal arm is {}'.format(np.round(arm, 4)))
        print('optimal reward is {:.3f}'.format(opt_reward))

        return {'reward': rewards}


def get_avg_sem(dicts):
    """ convert results from all trials in run_processes to mean and standard error of the mean
    """
    all_avg = {}
    all_sem = {}

    num_runs = len(dicts)

    # compute regret
    for dict in dicts:
        opt_reward = dict['optimal']['reward']
        for method in dict:
            dict[method]['regret'] = opt_reward - dict[method]['reward']

    # compute mean and sem
    for method in dicts[0]:
        all_avg[method] = {}
        all_sem[method] = {}
        for result in dicts[0][method]:
            all = np.array([dict[method][result] for dict in dicts])
            # all_avg[method][result] = np.median(all, axis=0)
            all_avg[method][result] = np.mean(all, axis=0)
            if result == 'regret':  # use regret sem for both
                all_sem[method]['regret'] = scipy.stats.sem(all, axis=0)
                all_sem[method]['reward'] = scipy.stats.sem(all, axis=0)

    return all_avg, all_sem


def run_single_process(seed):
    """ run single process """
    run = Run(N, T, synthetic, seed=seed, VERBOSE=False)

    run.get_historical(historical_T)

    result = {}

    if 'optimal' in models:
        print('\n------------------')
        print('optimal effort\n')
        out = run.optimal_effort()
        result['optimal'] = out
    if 'historical_exploit' in models:
        print('\n------------------')
        print('historical exploit\n')
        out = run.historical_exploit()
        result['historical_exploit'] = out
    if 'minion' in models:
        print('\n------------------')
        print('minion\n')
        out = run.minion(gamma=0.5, eta=3.1, beta_ftpl=4.2)
        result['minion'] = out
    if 'CUCB' in models:
        print('\n------------------')
        print('CUCB\n')
        out = run.CUCB()
        result['CUCB'] = out
    if 'random' in models:
        print('\n------------------')
        print('random\n')
        out = run.random(exploit=True)
        result['random'] = out
        result['random_exploit'] = {'reward': out['exploit_reward']}
    if 'lipschitz' in models:
        print('\n------------------')
        print('standard lipschitz\n')
        out = run.standard_lipschitz()
        result['lipschitz'] = out
    if 'eps_greedy' in models:
        print('\n------------------')
        print('epsilon-greedy\n')
        out = run.epsilon_greedy(eps=0.1)
        result['eps_greedy'] = out
    if 'decomposed' in models:
        print('\n------------------')
        print('regular decomposed\n')
        out = run.decomposed_lipschitz(increasingness=True, history=True, use_features=True, VERBOSE=False)
        result['decomposed'] = out
        result['decomposed_exploit'] = {'reward': out['exploit_reward']}

    print('optimal arm is {}'.format(np.round(run.optimal, 3)))

    return result


def run_processes():
    pool = Pool(processes=num_processes)

    # np.random.seed(42)
    trial_seeds = [np.random.randint(1000) for _ in range(num_repeats)]
    out = pool.map(run_single_process, trial_seeds)

    all_avg, all_sem = get_avg_sem(out)
    return all_avg, all_sem


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--targets', '-N', help='number of targets', type=int, default=25)
    parser.add_argument('--horizon', '-T', help='time horizon', type=int, default=500)
    parser.add_argument('--budget', '-B', help='patrol budget', type=int, default=1)
    parser.add_argument('--num_repeats', '-R', help='number of repeats', type=int, default=30)
    parser.add_argument('--num_processes', '-P', help='number of processes', type=int, default=4)
    parser.add_argument('--historical_T', '-H', help='number of historical timesteps', type=int, default=50)
    parser.add_argument('--verbose', '-V', help='if True, then verbose output (default False)', action='store_true')
    # parser.add_argument('--synthetic', '-S', help='if True, then use synthetic data; else real-world (default)', action='store_true')

    args = parser.parse_args()

    num_repeats   = args.num_repeats      # number of trials to run
    num_processes = args.num_processes  # how many processes to run in parallel (depends on number of cores)
    historical_T  = args.historical_T    # number of historical points
    N = args.targets    # num targets
    T = args.horizon    # num timesteps
    B = args.budget     # budget

    synthetic = True #args.synthetic
    VERBOSE = args.verbose

    models = ['optimal',
              'decomposed',
              'historical_exploit',
              'lipschitz',
              'CUCB',
              'minion',
              ]

    colors = {'optimal': 'k',
              'eps_greedy': 'pink',
              'historical_exploit': 'gold',
              'decomposed': 'cornflowerblue',
              'decomposed_exploit': 'darkblue',
              'lipschitz': 'darkorange',
              'minion': 'olive',
              'CUCB': 'red',

              'decomposed_features': 'turquoise',
              'decomposed_features_exploit': 'teal',
              'decomposed_no_increasing': 'red',
              'decomposed_history': 'limegreen',
              'decomposed_history_exploit': 'darkgreen',
              'decomposed_no_history': 'green',
              'random': 'lightsteelblue',
              'random_exploit': 'slategray',
              }

    # generate data
    num_features  = 4
    num_hids      = 100
    num_layers    = 10
    num_instances = 1       # num games
    num_samples   = 6

    plot_compare()
