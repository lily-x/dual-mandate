""" from end-to-end SSG code by Andrew Perrault """

import numpy as np
import torch
import math

from sklearn.tree import DecisionTreeClassifier

np.random.seed(42)

def generate_synthetic_data(num_targets, num_features, num_hids,
                       num_layers, num_instances, num_samples,
                       attacker_w=-4.0, noise=None):
    # add test instances
    # num_instances = num_instances + 50

    features = np.random.uniform(low=-10., high=10., size=(num_instances, num_targets, num_features))
    # attacker_w = np.random.uniform(low = -2., high=-0.)

    layer_list_attacker = []
    for i in range(num_layers):
        if i == 0:
            layer_list_attacker.append(torch.nn.Linear(num_features, num_hids))
        else:
            layer_list_attacker.append(torch.nn.Linear(num_hids, num_hids))
        layer_list_attacker.append(torch.nn.ReLU())
    layer_list_attacker.append(torch.nn.Linear(num_hids, 1))

    layer_list_defender = []
    for i in range(num_layers):
        if i == 0:
            layer_list_defender.append(torch.nn.Linear(num_features, num_hids))
        else:
            layer_list_defender.append(torch.nn.Linear(num_hids, num_hids))
        layer_list_defender.append(torch.nn.ReLU())
    layer_list_defender.append(torch.nn.Linear(num_hids, 1))

    attacker_model = torch.nn.Sequential(*layer_list_attacker)
    attacker_values = attacker_model(torch.from_numpy(features).float()).squeeze() * 10.
    if noise:
        attacker_values = attacker_values + torch.normal(mean=0.0, std=noise)
    # for numerical stability
    attacker_values = attacker_values - torch.max(attacker_values)

    defender_model = torch.nn.Sequential(*layer_list_defender)
    defender_values = defender_model(torch.from_numpy(features).float()).squeeze()
    # defender values are negative
    defender_values = defender_values - torch.max(defender_values)
    defender_values = defender_values / -torch.min(defender_values)

    #defender_values = torch.from_numpy(np.ones_like(attacker_values.detach().numpy()) * -1.)

    # print(defender_values)
    # print(attacker_values)
    return (features / math.sqrt(1. / 12. * 20 ** 2), defender_values.detach().numpy(), attacker_values.detach().numpy(), attacker_w)


if __name__ == '__main__':
    num_targets   = 25
    num_features  = 3
    num_hids      = 100
    num_layers    = 10
    num_instances = 1
    num_samples   = 6
    data = generate_synthetic_data(num_targets, num_features, num_hids,
                           num_layers, num_instances, num_samples,
                           attacker_w=-4.0)

    # print(data)
    features = data[0].squeeze()
    defender_vals = data[1]
    attacker_vals = data[2]
    attacker_w = data[3]

    print(features)
    print('---')
    print(defender_vals)
    print('---')
    print(attacker_vals)
    print('---')
    print(attacker_w)


    # simulate historical data of attacks
    targets = range(num_targets)
    attack_range = (2, 7)
    attacker_p = (-attacker_vals) / (-attacker_vals).sum()
    historical_t = 5
    print()
    print(attacker_p)

    attacks = np.zeros((historical_t, num_targets))
    for i in range(historical_t):
        num_attacks = np.random.randint(*attack_range)
        curr_attacks = np.random.choice(targets, num_attacks, p=attacker_p)
        print(curr_attacks)

        # make attacks binary
        attacks[i, np.unique(curr_attacks)] = 1

    print(attacks)


    # learn an ML model
    inputs = np.tile(features, (historical_t, 1))
    # time   = np.repeat(range(historical_t), num_targets).reshape(-1, 1)
    # inputs = np.hstack((time, inputs))

    labels = attacks.flatten()

    classifier = DecisionTreeClassifier()
    classifier.fit(inputs, labels)

    predictions = classifier.predict_proba(features)
    predictions = predictions[:, 1]  # prediction of a positive label
    print(predictions)
    print(np.round(attacker_p, 2))


    # compute mean absolute error
    mean_abs_error = np.mean(np.abs(attacker_p - predictions))
    print('mean abs error: {:.5f}'.format(mean_abs_error))
