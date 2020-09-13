""" Lily Xu

implement discrete version of zooming algorithm for Lipschitz bandits
"""

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm

np.random.seed(42)

out_path = './plots'
VISUALIZE = False

class DiscreteLipschitz:
    """ implement Lipschitz bandits (discrete space)
    (Kleinberg, Slivkins, Upfal 2013) """
    def __init__(self, targets, adversary, T=1, alpha=None, adv_type='effort', VERBOSE=False):
        """
        params
        ------
        targets (num_targets, num_features)
        """
        self.VERBOSE = VERBOSE

        self.T = T              # num phases
        self.adversary = adversary

        # scale to diameter <= 1
        if adv_type in ['stochastic', 'recharging']:
            # scale each arms array to range [0, 1]
            targets -= np.min(targets, axis=0)
            targets /= np.max(targets, axis=0)

            # diameter of P is still bigger than 1...
            num_dim = targets.shape[1]
            targets /= np.sqrt(num_dim)


        self.targets = targets  # targets with features
        num_targets = targets.shape[0]

        self.AA = set(range(num_targets))  # all arms, just ID

        # define euclidean distances between targets
        d = np.zeros((num_targets, num_targets))
        for a in range(num_targets):
            for b in range(num_targets):
                diff = targets[a, :] - targets[b, :]
                d[a, b] = np.sqrt(np.sum(np.square(diff)))

        self.d = d             # distances

        self.n = {}
        self.cum_rewards = {}

        self.i = 0                 # current phase

        # reset reward and arm pulls
        for a in self.AA:
            self.n[a] = 0
            self.cum_rewards[a] = 0

        # set all arms as inactive
        self.U = self.AA.copy()      # uncovered arms
        self.A = set()               # active arms

        if adv_type in ['recharging', 'recharging_coverage']:
            assert alpha is not None
            self.alpha = alpha
            if adv_type == 'recharging':
                # for recharging adversary
                self.elapsed_time = np.zeros(num_targets)
            elif adv_type == 'recharging_coverage':
                for target in self.targets:
                    assert self.arm_valid(target)

        self.adv_type = adv_type


    def arm_valid(self, x):
        """ returns whether arm x is valid """
        x = np.array(x)
        max_val = 1 / (1 + self.alpha)

        # test within bounds
        if np.any(x < 0) or np.any(x > max_val):
            if self.VERBOSE:
                print('  ERROR not within bounds: x {}, max val {}'.format(x, max_val))
            return False

        # test sum of beta
        # if np.abs(1 - np.sum(x / (1 - alpha*x))) > 1e-7:
        beta_sum = np.sum(x / (1 - self.alpha*x))
        if beta_sum > 1 + 1e-7:
            if self.VERBOSE:
                print('  ERROR beta sum exceeds 1! sum={}'.format(beta_sum))
            return False
        else:
            if self.VERBOSE:
                print('beta sum {:.3f}'.format(beta_sum))

        return True


    def zoom_step(self, display=False):
        """ execute single step of zooming algorithm
        returns arm that was selected """
        if len(self.U) > 0:
            # print('arm inactive')
            self.activation_rule()

        x, reward = self.selection_rule()

        if display:
            print('active arms', self.A)
            print('uncovered arms', self.U)
            for a in self.A:
                print('  {} radius {:.3f}, mu {:.3f}, n {}'.format(a, self.r(a),
                        self.cum_rewards[a] / max(1, self.n[a]), self.n[a]))

        return x, reward


    def zooming(self):
        all_reward = np.zeros(self.T)

        """ zooming algorithm
        execute entire process """
        # for i in range(1, self.T + 1):
        for i in [1]:
            self.i = i

            # reset reward and arm pulls
            for a in self.AA:
                self.n[a] = 0
                self.cum_rewards[a] = 0

            # set all arms as inactive
            self.U = self.AA.copy()      # uncovered arms
            self.A = set()               # active arms

            for t in range(self.T):
                if t % 10 == 0 and self.VERBOSE:
                    x, reward = self.zoom_step(display=True)
                    print(' phase {}, round {}, arm = {}'.format(i, t, x))
                else:
                    x, reward = self.zoom_step()

                all_reward[t] = reward

            if VISUALIZE:
                self.visualize(i)

        return all_reward


    def activation_rule(self):
        """ activation rule """
        # select arm x at random from U
        x = np.random.choice(tuple(self.U))
        self.U.remove(x)
        self.A.add(x)

    def selection_rule(self):
        """ selection rule
        returns arm that was selected """
        assert len(self.A) > 0

        # let x be the arm in A with maximal index(x, t)
        x = -1
        max_index = -1
        for a in self.A:
            index = self.index(a)
            if index > max_index:
                x = a
                max_index = index

        # print('max index {}'.format(max_index))
        assert x != -1

        # visit target x
        self.n[x] += 1
        if self.adv_type == 'effort':
            reward = 0
            for i, eff in enumerate(self.targets[x]):
                reward += self.adversary.visit_target(i, eff)
            self.cum_rewards[x] += reward

        else:
            reward = self.adversary.visit_target(x)
            self.cum_rewards[x] += reward

        # check which arms are currently uncovered
        self.U = self.AA.copy()
        for a in self.A:
            for b in self.AA:
                if self.d[b, a] <= self.r(a):
                    self.U.discard(b)

        if self.adv_type == 'recharging':
            self.elapsed_time += 1
            self.elapsed_time[x] = 0

        return x, reward


    def index(self, x):
        """ similar to index in UCB1 algorithm
        mu = sample average
        r = radius
        """
        if self.adv_type == 'recharging':
            mu = self.elapsed_time[x] / (self.elapsed_time[x] + self.alpha[x])
            print('arm {:2.0f}, elapsed {:2.0f}, alpha {:2.2f}, mu {:.2f}'.format(x, self.elapsed_time[x], self.alpha[x], mu))
            return mu
        else:
            mu = self.cum_rewards[x] / max(self.n[x], 1)

        return mu + 2 * self.r(x)


    def r(self, x):
        """ confidence radius
        i = phase
        n = num pulls """

        r = np.sqrt( (8 * self.i) / (1 + self.n[x]))
        # r = np.sqrt( (8 * self.i) / (1 + 200 * self.n[x]))
        return r


    def visualize(self, phase=''):
        fig, ax = plt.subplots()
        plt.axis('scaled')
        ax.set_xlim(np.min(self.targets[:, 0]), np.max(self.targets[:, 0]))
        ax.set_ylim(np.min(self.targets[:, 1]), np.max(self.targets[:, 1]))

        center_r = .01

        # plot all arms
        for a in self.AA:
            center = plt.Circle((self.targets[a, 0], self.targets[a, 1]), .5 * center_r, color='r')
            ax.add_artist(center)

        # plot active arms
        for a in self.A:
            coords = (self.targets[a, 0], self.targets[a, 1])

            # plot confidence radius
            # color corresponds to mu
            mu = self.cum_rewards[a] / max(1, self.n[a])
            mu = max(0, mu)  # prevent negative
            mu = min(1, mu)  # prevent > 1
            circle = plt.Circle(coords, self.r(a), color=cm.winter(mu), alpha=.5)

            # plot center
            center = plt.Circle(coords, center_r, color='k', alpha=.7)

            ax.add_artist(circle)
            ax.add_artist(center)

            # text?

            plt.plot()

        # plt.show()
        fig.savefig('{}/plot_lipschitz_phase_{}.png'.format(out_path, phase))
        plt.close(fig)


if __name__ == '__main__':
    T = 1 # num phases
    width = 3
    height = 3
    num_targets = width * height
    num_features = 2

    mu    = np.random.uniform(size=num_targets)
    sigma = np.random.uniform(size=num_targets)

    print('\n')
    print('mu')
    for i, m in enumerate(mu):
        print('  i {}, {:.3f}'.format(i, m))
    print('\n')

    # reduce each dimension down
    targets = np.zeros((num_targets, num_features))
    id = 0
    for i in range(width):
        for j in range(height):
            targets[id][0] = i
            targets[id][1] = j
            id += 1

    print(targets)
    print(targets.shape)

    lipschitz = Lipschitz(targets, mu, sigma, T)
    lipschitz.zooming()
