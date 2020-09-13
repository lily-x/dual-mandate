""" analyze the impact of historical data and why we could be misled early on """

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import scipy

from baseline import smooth

out_path = './output'

def main():
    N             = 100
    n_historical  = 2000
    n_top         = 10  # num top values
    n_trials      = 30  # num repeats
    timestamp     = datetime.now()


    avg_empirical = np.zeros((n_trials, n_historical)) # avg empirical reward of selected top targets
    avg_true      = np.zeros((n_trials, n_historical)) # true avg reward of selected top targets
    avg_all       = np.ones((n_trials, n_historical)) # true avg value of all targets
    avg_top_true  = np.ones((n_trials, n_historical)) # avg value of true top targets

    # run multiple trials
    for trial in range(n_trials):
        true_rewards  = np.random.random(N)

        true_top               = true_rewards.argsort()[-n_top:]
        avg_top_true[trial, :] = np.mean(true_rewards[true_top])
        avg_all[trial, :]      = np.mean(true_rewards)

        rewards       = np.zeros(N)
        n_pulls       = np.zeros(N)

        for t in range(n_historical):
            # select a target at random
            i = np.random.randint(N)

            # observe reward
            reward = np.random.binomial(1, true_rewards[i])
            rewards[i] += reward
            n_pulls[i] += 1

            avg_rewards = rewards / n_pulls
            avg_rewards[n_pulls == 0] = 0  # avoid nan in divide by zero

            top = avg_rewards.argsort()[-n_top:]

            # print(t, avg_empirical)

            avg_empirical[trial, t] = avg_rewards[top].mean()
            avg_true[trial, t]      = true_rewards[top].mean()

    # average over trials
    empirical_avg = np.mean(avg_empirical, axis=0)
    true_avg      = np.mean(avg_true, axis=0)
    all_avg       = np.mean(avg_all, axis=0)
    top_true_avg  = np.mean(avg_top_true, axis=0)

    empirical_sem = scipy.stats.sem(avg_empirical, axis=0)
    true_sem      = scipy.stats.sem(avg_true, axis=0)
    all_sem       = scipy.stats.sem(avg_all, axis=0)
    top_true_sem  = scipy.stats.sem(avg_top_true, axis=0)

    true_avg = smooth(true_avg, weight=0.93)


    x_vals = np.arange(n_historical)


    fig = plt.figure()

    plt.plot(x_vals, empirical_avg, label='empirical top {} - empirical reward'.format(n_top), c='purple')
    plt.fill_between(x_vals, empirical_avg-empirical_sem, empirical_avg+empirical_sem, facecolor='purple', alpha=0.15)
    plt.plot(x_vals, true_avg, label='empirical top {} - true reward'.format(n_top), c='blue')
    plt.fill_between(x_vals, true_avg-true_sem, true_avg+true_sem, facecolor='blue', alpha=0.15)
    plt.plot(x_vals, all_avg, label='true top {} - true reward'.format(n_top), c='orange')
    plt.fill_between(x_vals, all_avg-all_sem, all_avg+all_sem, facecolor='orange', alpha=0.15)
    plt.plot(x_vals, top_true_avg, label='all targets - true reward', c='green')
    plt.fill_between(x_vals, top_true_avg-top_true_sem, top_true_avg+top_true_sem, facecolor='green', alpha=0.15)

    plt.title('historical data N={}'.format(N))
    plt.xlabel('num historical points')
    plt.ylabel('avg reward')
    plt.legend()
    fig.tight_layout()
    plt.savefig('{}/historical_data_N{}_top{}_{}.png'.format(out_path, N, n_top, timestamp))
    plt.show()


    # save to CSV
    reward_dict = {'avg_empirical': empirical_avg, 'sem_empirical': empirical_sem, 'avg_true': true_avg, 'avg_top_true': top_true_avg, 'avg_all': all_avg}
    reward_df = pd.DataFrame(reward_dict)
    reward_df.to_csv('{}/historical_data_N{}_top{}_{}.csv'.format(out_path, N, n_top, timestamp))



if __name__ == '__main__':
    main()
