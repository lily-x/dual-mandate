""" Lily Xu
January 2020
different adversarial models """

import abc
import numpy as np

class Adversary(abc.ABC):
    @abc.abstractmethod
    def visit_target(self, x):
        pass



class PWLFunction():
    def __init__(self, slope, breakpoints):
        # breakpoints should be one longer than slope (start and end of each segment)
        assert len(slope) == len(breakpoints) - 1

        # ensure function begins at 0
        assert breakpoints[0] == 0

        self.slope = slope
        self.breakpoints = breakpoints

        self.seg_widths = np.roll(self.breakpoints, -1) - self.breakpoints
        self.seg_widths = self.seg_widths[:-1]

    def get_reward(self, val):
        # TODO: ensure we never keep getting more reward for bigger effort

        if val >= self.breakpoints[-1]:
            # we're past the last segment, so set reward to end of last seg
            reward = np.sum(self.slope * self.seg_widths)
        else:
            active_seg = np.where(val < self.breakpoints)[0][0] - 1

            reward = np.sum(self.slope[:active_seg] * self.seg_widths[:active_seg]) + \
                    self.slope[active_seg] * (val - self.breakpoints[active_seg])

        return reward

    def get_y(self):
        y = [self.get_reward(x) for x in self.breakpoints]
        return np.array(y)



class EffortAdversary(Adversary):
    def __init__(self, alpha):
        self.N = len(alpha)
        # self.alpha = alpha

        # make PWL reward for each target
        self.num_seg = 10
        self.pwl = {}
        for i in range(self.N):
            self.pwl[i] = self.generate_pwl_reward()

            assert self.pwl[i].get_reward(1.) <= 1.

        start_x, end_x, start_y, end_y, slope = self.get_start_end()
        for i in range(self.N):
            assert end_y[i][-1] <= 1.

    def visit_target(self, target, effort):
        # rare issue with very small negative number
        if -1e-6 < effort < 0:
            effort = 0

        assert 0. <= effort <= 1. + 1e-4, 'effort {} not within 0 and 1'.format(effort)
        # reward = np.log(effort / self.alpha[target] + 1)
        # reward = effort / (effort + self.alpha[target])  # concave reward

        # x = effort / self.alpha[target]
        # reward = ((2*x-2)**3 + 2*(x-2)**2) / 12.    # not increasing
        reward = self.pwl[target].get_reward(effort)

        return reward


    def generate_pwl_reward(self):
        """ randomly generate increasing, PWL reward function """
        # features are correlated based on similarity of the PWL reward functions

        # TODO: fix to 0 - 1
        breakpoints = np.linspace(0, 1, self.num_seg+1)
        slope = np.random.uniform(size=self.num_seg)
        # slope = np.flip(np.sort(slope))       # produce concave function

        assert slope.sum() * (1. / self.num_seg) <= 1.

        pwl = PWLFunction(slope, breakpoints)

        return pwl


    def get_start_end(self):
        start_x = [self.pwl[i].breakpoints[:-1] for i in range(self.N)]
        end_x = [self.pwl[i].breakpoints[1:] for i in range(self.N)]
        # for i in range(self.N):
        #     # ends = np.roll(self.pwl[i].breakpoints, -1)
        #     # ends = ends[:-1]
        #     end_x.append(ends)

        start_y = [[self.pwl[i].get_reward(x) for x in start_x[i]] for i in range(self.N)]
        end_y   = [[self.pwl[i].get_reward(x) for x in end_x[i]] for i in range(self.N)]
        slope   = [self.pwl[i].slope for i in range(self.N)]

        return start_x, end_x, start_y, end_y, slope


    def compute_pwl_distance(self, i1, i2):
        """ given two PWL functions, compute the distance between the two functions
        in feature space (ignore effort space; that is addressed separately)

        this is used as our lipschitz constant L for feature space"""

        # ensure breakpoints are the same
        np.testing.assert_array_equal(self.pwl[i1].breakpoints, self.pwl[i2].breakpoints)

        y1 = self.pwl[i1].get_y()
        y2 = self.pwl[i2].get_y()

        max_dist = np.max(np.abs(y1 - y2))
        return max_dist

class RealEffortAdversary(EffortAdversary):
    def __init__(self, N, pos=0):
        assert N in {25, 100}

        print('using real-world effort! pos {}'.format(pos))

        import pickle
        if N == 25:
            input_file = 'input/predicted_probabilities_5_40.pickle'
        elif N == 100:
            input_file = 'input/predicted_probabilities_10_40.pickle'

        with open(input_file, 'rb') as f:
            data = pickle.load(f)

        data = data[pos, :, :]

        self.N = data.shape[0]
        self.num_seg = data.shape[1] - 1

        self.pwl = {}

        breakpoints = np.linspace(0, 1., self.num_seg + 1)

        # set 0 eff = 0 reward
        data[:, 0] = 0

        for i in range(self.N):
            slope = np.zeros(self.num_seg)

            # prevent decreasing
            for b in range(1, self.num_seg + 1):
                if data[i, b] < data[i, b - 1]:
                    data[i, b] = data[i, b - 1]

            for b in range(self.num_seg):
                slope[b] = (data[i, b + 1] - data[i, b]) / (1. / self.num_seg)

            self.pwl[i] = PWLFunction(slope, breakpoints)

            assert self.pwl[i].get_reward(1.) <= 1.

        start_x, end_x, start_y, end_y, slope = self.get_start_end()
        for i in range(self.N):
            assert end_y[i][-1] <= 1.
