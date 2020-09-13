""" Lily Xu

LIZARD algorithm
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB

import matplotlib.pyplot as plt
import matplotlib.cm as cm


def solve_exploit(B, n, cum_rewards, budget):
    """ given historical data, solve for optimal arm pull

    used in to exploit in random baseline, epsilon-greedy """
    N = len(B)
    model = gp.Model('exploit')

    # compute mu values
    mu = [{} for _ in range(N)]
    for i in range(N):
        for j, eff in enumerate(B[i]):
            mu[i][eff] = cum_rewards[i][eff] / max(1, n[i][eff])

    # silence output
    model.setParam('OutputFlag', 0)

    x = [[model.addVar(vtype=GRB.BINARY, name='x_{}_{}'.format(i, j))
            for j in range(len(B[i]))] for i in range(N)]

    model.setObjective(gp.quicksum([x[i][j] * mu[i][eff]
                for i in range(N) for j, eff in enumerate(B[i])]),
                GRB.MAXIMIZE)

    model.addConstrs((gp.quicksum(x[i][j] for j, eff in enumerate(B[i])) == 1
                for i in range(N)), 'one_per_target') # pull one arm per target

    model.addConstr(gp.quicksum([x[i][j] * B[i][j]
                for i in range(N) for j, eff in enumerate(B[i])]) <= budget, 'budget')  # stay in budget

    model.optimize()

    if model.status != GRB.OPTIMAL:
        raise Exception('Uh oh! Model status is {}'.format(model.status))

    # convert x to beta
    exploit_arm = np.full(N, np.nan)
    for i in range(N):
        for j in range(len(B[i])):
            if abs(x[i][j].x - 1) < 1e-2:
                exploit_arm[i] = B[i][j]

        assert not np.isnan(exploit_arm[i]), 'MIP x[{}] vals are {}'.format(i, [x[i][j].x for j in range(len(B[i]))])

    return exploit_arm


class DecomposedLipschitz:
    """ implement Lipschitz bandits (with decomposed rewards) """
    def __init__(self, targets, adversary, T, optimal, budget,
                increasingness=True, history=None, use_features=False,
                VERBOSE=False):
        """
        params
        ------
        targets (num_targets, num_features)
        """

        self.VERBOSE = VERBOSE

        self.adversary = adversary
        self.targets   = targets    # targets with features

        self.optimal = optimal
        self.increasingness = increasingness  # whether we want to build in that reward function is increasing - monotonicity
        self.use_features = use_features

        self.N = targets.shape[0]   # num targets
        self.T = T                  # num timesteps

        self.budget = budget

        self.B           = [[] for _ in range(self.N)]  # list of active arms (decomposed)
        self.cum_rewards = [{} for _ in range(self.N)]  # cumulative reward
        self.n           = [{} for _ in range(self.N)]  # number of pulls

        self.n_w_history = [{} for _ in range(self.N)]  # number of pulls INCL historical data

        self.L = np.ones(self.N)  # Lipschitz constant in each dimension

        self.t = 0 # current timestep

        # initialize with historical data
        num_discretization = 10
        eff_levels = np.linspace(0, 1, num_discretization+1)
        # eliminating floating point glitches
        eff_levels = np.round(eff_levels, 3)
        self.eff_levels = eff_levels

        for i in range(self.N):
            for eff in eff_levels:
                if eff not in self.n[i]:
                    self.B[i].append(eff)
                    self.n[i][eff] = 0
                    self.n_w_history[i][eff] = 0
                    self.cum_rewards[i][eff] = 0

        if history is not None:
            for i in range(len(history['B'])):
                for j, eff in enumerate(history['n'][i]):
                    eff = np.round(eff, 3)
                    self.n[i][eff] = 0
                    self.n_w_history[i][eff] = history['n'][i][eff]
                    self.cum_rewards[i][eff] = history['cum_rewards'][i][eff]

        # integrate features by computing distance in feature space
        # (== distance in reward functions)
        self.dist = np.zeros((self.N, self.N))
        if self.use_features:
            for i1 in range(self.N):
                for i2 in range(self.N):
                    if i1 == i2: continue
                    self.dist[i1, i2] = self.adversary.compute_pwl_distance(i1, i2)

        self.t_uncover = np.full(self.T, np.nan)  # track which arms were uncovered at each timestep
        self.t_ucb = np.full(self.T, np.nan)
        self.exploit_rewards = np.zeros(self.T)
        self.true_reward = np.zeros(self.T)
        self.mip_UCB = np.zeros(self.T)


    def activation_rule(self):
        """ return an uncovered arm if one exists, or None if space is covered """
        beta = None

        if self.t == 0:
            #### discretization
            num_random = 10

            # make random arms in a grid to initialize
            for i in range(num_random):
                arms = np.linspace(0, 1, num_random+1)
                make_arms = np.ones(self.N) * arms[i]

                self.create_arms(make_arms)


        # initialize if doesn't exist
        if beta is not None:
            self.create_arms(beta)

        return beta


    def zoom_step(self, display=False):
        """ execute single step of zooming algorithm
        returns arm that was selected """

        beta, exploit_beta = self.selection_rule()

        # pull arm
        rewards = np.zeros(self.N)
        for i in range(self.N):
            eff = beta[i]

            # get reward, with optimistic bound based on distance from arm center
            reward_prob = self.adversary.visit_target(i, beta[i])
            observed_reward = np.random.choice([0, 1], p=[1-reward_prob, reward_prob])

            rewards[i] = reward_prob

            self.n_w_history[i][eff] += 1
            self.n[i][eff] += 1
            self.cum_rewards[i][eff] += observed_reward

            # get reward of exploit
            self.exploit_rewards[self.t] += self.adversary.visit_target(i, exploit_beta[i])

        self.true_reward[self.t] = rewards.mean()
        if self.VERBOSE:
            print('             true reward {:.4f}'.format(self.true_reward[self.t]))

        return beta, rewards


    def zooming(self):
        """ zooming algorithm
        execute entire process """

        all_beta   = np.zeros((self.T, self.N))
        all_reward = np.zeros((self.T, self.N))

        for t in range(self.T):
            zoom_display = True if self.VERBOSE and t % 10 else False

            beta, reward = self.zoom_step(display=zoom_display)

            all_reward[t, :] = reward
            all_beta[t, :] = beta

            if self.VERBOSE and zoom_display:
                print(' round {}, beta = {}'.format(t, beta))

            self.t += 1

        all_reward = np.sum(all_reward, axis=1)

        if self.VERBOSE:
            print('\nbeta')
            for t in range(self.T):
                print('  {} {} {:.3f}'.format(t, np.round(all_beta[t, :], 3), all_beta[t, :].sum()))

            print('\nreward per pull')
            for i in range(self.N):
                print('----')
                print('  {}'.format(i))
                for eff in sorted(self.cum_rewards[i].keys()):
                    true_mu = self.adversary.pwl[i].get_reward(eff)
                    mu = self.cum_rewards[i][eff] / max(self.n_w_history[i][eff], 1)
                    print('    n {:3.0f}, eff {:.4f}, mu {:.3f}, true mu {:.3f}, conf {:.3f}'.format(
                        self.n_w_history[i][eff], eff, mu, true_mu, self.conf(i, eff)))

            percent_wrong = (len(np.where(self.mip_UCB < self.true_reward)[0]) / self.T) * 100
            print('\nUCB wrong {:.2f}% of the time'.format(percent_wrong))

        return all_reward


    def selection_rule(self):
        """ selection rule
        returns arm that was selected

        budget b

            # GIVEN VARIABLES
            # B[i]
            # y in B[i]
            # L[i]
            # r[i](y)
            # mu[i](y)
            # M
        """
        M = 1e6    # big M

        model = gp.Model('milp')


        # silence output
        model.setParam('OutputFlag', 0)

        MIP_type = 'max_budget'

        if MIP_type == 'i_small':
            ns = {}
            mus = {}

            for i in range(self.N):
                for j, eff in enumerate(self.B[i]):
                    if self.n_w_history[i][eff] == 0:
                        mu = 1. / 2.
                    else:
                        mu = self.cum_rewards[i][eff] / self.n_w_history[i][eff]

                    ns[(i,j)] = self.n_w_history[i][eff]
                    mus[(i,j)] = mu


            # w: auxiliary variable = x_ij * I_small
            w    = [[model.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY, name='w_{}_{}'.format(i, j))
                    for j in range(len(self.B[i]))] for i in range(self.N)]

            # x: indicator saying pull arm j at target i and use Hoeffing bound
            x    = [[model.addVar(vtype=GRB.BINARY, name='x_{}_{}'.format(i, j))
                    for j in range(len(self.B[i]))] for i in range(self.N)]

            # I_small: indicator saying arm pulled constributing min num samples to the Hoeffing bound
            I_small = [model.addVar(vtype=GRB.BINARY, name='I_small_{}'.format(i))
                       for i in range(self.N)]

            model.setObjective(gp.quicksum([x[i][j] * mus[(i,j)]
                for i in range(self.N) for j, eff in enumerate(self.B[i])]) /
                self.N +
                gp.quicksum([w[i][j] * self.generic_r(self.N * ns[(i,j)])
                    for i in range(self.N) for j, eff in enumerate(self.B[i])]),
                GRB.MAXIMIZE)

            model.addConstrs((gp.quicksum(x[i][j] for j, eff in enumerate(self.B[i])) == 1
                                for i in range(self.N)), 'one_per_target') # pull one arm per target

            model.addConstr(gp.quicksum([x[i][j] * self.B[i][j]
                                for i in range(self.N) for j, eff in enumerate(self.B[i])]) <= self.budget, 'budget')  # stay in budget

            model.addConstrs((-M * (1 - I_small[i]) +
                            gp.quicksum([x[i][j] * ns[(i,j)] for j, eff in enumerate(self.B[i])]) <=
                            gp.quicksum([x[k][j] * ns[(k,j)] for j, eff in enumerate(self.B[k])])
                            for i in range(self.N) for k in range(self.N)), 'big_thing')

            model.addConstr(gp.quicksum(I_small) == 1, 'only_one_i_small')

            model.addConstrs(w[i][j] <= x[i][j]
                            for i in range(self.N) for j, eff in enumerate(self.B[i]))

            model.addConstrs((w[i][j] <= I_small[i]
                            for i in range(self.N) for j, eff in enumerate(self.B[i])), 'wi_constr')

            model.optimize()

            opt_ns = np.zeros(self.N)
            opt_arm_ucb = 0
            for i in range(self.N):
                for j, eff in enumerate(self.B[i]):
                    if abs(eff - self.optimal[i]) < 1e-4:
                        opt_arm_ucb += mus[(i,j)]
                        opt_ns[i] = ns[(i,j)]

            opt_arm_ucb /= self.N
            opt_arm_ucb += self.generic_r(self.N * np.min(opt_ns))


            self.mip_UCB[self.t] = opt_arm_ucb

        elif MIP_type == 'max_budget':

            pre_index = {}
            index = {}

            # compute pre-indexes
            for i in range(self.N):
                for j, eff in enumerate(self.B[i]):
                    eff = self.B[i][j] # keep in case of floating point error

                    if eff == 0:
                        mu = 0.
                    elif self.n_w_history[i][eff] == 0:
                        mu = 1.
                    else:
                        mu = self.cum_rewards[i][eff] / self.n_w_history[i][eff]

                    conf = self.conf(i, eff)
                    pre_index[(i,j)] = mu + conf

            use_pre_index = {}


            # compute indexes - with feature distance
            for i1 in range(self.N):
                for j1, eff1 in enumerate(self.B[i1]):
                    eff1 = self.B[i1][j1]  # used to prevent floating point issues

                    use_pre_index[(i1, j1)] = '-'

                    # monotonicity: zero equals zero assumption
                    # with 0 effort == 0 reward assumption, set uncertainty to 0
                    if self.increasingness:
                        if eff1 == 0:
                            index[(i1, j1)] = 0.
                            continue

                    min_pre = pre_index[(i1, j1)]

                    if self.use_features:
                        loop_over = range(self.N)
                    else:
                        loop_over = [i1]

                    for i2 in loop_over:
                        for j2, eff2 in enumerate(self.B[i1]):
                            eff2 = self.B[i2][j2]  # used to prevent floating point issues

                            if self.increasingness:
                                dist = max(0, eff1 - eff2) * self.L[i1]
                            else:
                                dist = abs(eff1 - eff2) * self.L[i1]
                            influenced_dist = pre_index[(i2, j2)] + dist + self.dist[i1, i2]
                            if influenced_dist < min_pre:
                                min_pre = influenced_dist
                                if abs(j1 - j2) > 1e-1:       # why does equality fail on these two ints??
                                # if j1 != j2:
                                    use_pre_index[(i1, j1)] = (i1, j2)
                                if abs(i1 - i2) > 1e-1:
                                    use_pre_index[(i1, j1)] = '{} @@@@@@'.format((i2, j2))
                                else:
                                    if min_pre == 0:
                                        print('weird! j1 {}, j2 {}, eff1 {:.2f}, eff2 {:.2f} dist {:.2f}'.format(j1, j2, eff1, eff2, dist))

                    index[(i1, j1)] = min_pre

            # x: indicator saying pull arm j at target i
            x = [[model.addVar(vtype=GRB.BINARY, name='x_{}_{}'.format(i, j))
                    for j in range(len(self.B[i]))] for i in range(self.N)]

            model.setObjective(gp.quicksum([x[i][j] * index[(i,j)]
                for i in range(self.N) for j in range(len(self.B[i]))]), GRB.MAXIMIZE)

            model.addConstrs((gp.quicksum(x[i][j] for j in range(len(self.B[i]))) == 1
                                for i in range(self.N)), 'one_per_target') # pull one arm per target

            model.addConstr(gp.quicksum([x[i][j] * self.B[i][j]
                                for i in range(self.N) for j in range(len(self.B[i]))]) <= self.budget, 'budget')  # stay in budget

            model.optimize()

            opt_arm_ucb = 0
            for i in range(self.N):
                for j, eff in enumerate(self.B[i]):
                    if abs(eff - self.optimal[i]) < 1e-4:
                        opt_arm_ucb += index[(i,j)]

            opt_arm_ucb /= self.N
            self.mip_UCB[self.t] = opt_arm_ucb

        if model.status != GRB.OPTIMAL:
            raise Exception('Uh oh! Model status is {}'.format(model.status))


        opt_reward = 0
        for i in range(self.N):
            opt_reward += self.adversary.pwl[i].get_reward(self.optimal[i])

        opt_reward /= self.N

        if self.VERBOSE:
            if MIP_type == 'i_small':
                print(' --- round {:4.0f}, arm UCB {:.3f}, opt arm UCB {:.3f}, opt_reward {:.3f}'.format(self.t, model.objVal / self.N, opt_arm_ucb, opt_reward))
            elif MIP_type == 'max_budget':
                print(' --- round {:4.0f}, arm UCB {:.3f}, opt arm UCB {:.3f}, opt_reward {:.3f}'.format(self.t, model.objVal, opt_arm_ucb, opt_reward))

        print_pulls = ''
        print_zero_pulls = ''

        for i in range(self.N):
            for j, eff in enumerate(self.B[i]):
                eff = self.B[i][j]

                # put * next to arms we pull
                star = '*' if x[i][j].x == 1 else ' '

                # put ! next to any UCBs with violations (UCB lower than true mu)
                true_mu = self.adversary.pwl[i].get_reward(eff)
                star2 = '!' if true_mu > index[(i,j)] else ' '

                n    = self.n_w_history[i][eff]
                mu   = self.cum_rewards[i][eff] / max(1, n)
                conf = self.conf(i, eff)

                out = '({:2.0f}, {:2.0f}) n {:3.0f}, eff {:.4f}, mu {:.3f}, true mu {:.3f}, conf {:.3f}, pre-I {:.3f}, I {:.3f} || {} {} {}'.format(
                                    i, j, n, eff, mu, true_mu, conf,
                                    pre_index[(i,j)], index[(i,j)],
                                    star, use_pre_index[(i,j)], star2)

                if n == 0:
                    print_zero_pulls += out + '\n'
                else:
                    print_pulls += out + '\n'

        if self.VERBOSE:
            print(print_pulls)
            print(print_zero_pulls)

        # for v in model.getVars():
        #     print('%s %g' % (v.varName, v.x))

        self.t_ucb[self.t] = model.objVal

        arm = np.full(self.N, np.nan)

        # convert x to beta
        for i in range(self.N):
            for j, eff in enumerate(self.B[i]):
                if abs(x[i][j].x - 1) < 1e-2:
                    arm[i] = self.B[i][j]

            assert not np.isnan(arm[i]), 'MIP x[{}] vals are {}'.format(i, [x[i][j].x for j in range(len(self.B[i]))])

        exploit_arm = solve_exploit(self.B, self.n_w_history, self.cum_rewards, self.budget)

        return arm, exploit_arm


    def generic_r(self, num_pulls):
        eps = .1
        r = np.sqrt(-np.log(eps) / (2. * max(1, num_pulls)))        # Andrew derivation

        return r

    def conf(self, i, eff):
        """ confidence radius of a given arm
        i = target
        eff = effort """
        assert eff <= 1.

        if eff in self.n[i]:
            num_pulls = self.n[i][eff]
        else:
            num_pulls = 0

        r = self.generic_r(num_pulls)

        return r
