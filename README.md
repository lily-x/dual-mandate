# Dual-mandate patrols

This code implements and evaluates algorithms for "[Dual-Mandate Patrols: Multi-Armed Bandits for Green Security](https://arxiv.org/abs/2009.06560)," including the LIZARD algorithm provided in the paper.

```
@inproceedings{xu2020dual,
  title={Dual-Mandate Patrols: Multi-Armed Bandits for Green Security},
  author={Xu, Lily and Bondi, Elizabeth and Fang, Fei and Perrault, Andrew and Wang, Kai and Tambe, Milind},
  booktitle={arXiv:2009.06560},
  year={2020}
}
```

## Files

`baseline.py` executes code and makes graphs (Figure 5 of paper)

`decomposed_zooming.py` is the LIZARD we propose, integrating Lipschitz-continuity and decomposition

`discrete_lipschitz.py` is a discrete version of zooming algorithm for Lipschitz bandits

`MINION.py` implements MINION algorithm from Gholami et al. [AAMAS 2019]

`adversary.py` specifies the behavior of the adversary

`historical_data_plot.py` produces plot of why relying on historical data is unreliable in early rounds (Figure 2 of paper)


## Data
We provide code to generate synthetic data with the function `get_historical()` in `baseline.py`. We cannot provide real-world data used in the paper due to the sensitive nature of poaching data.


## Execution
Execute code using `baseline.py`. The default setting is
> python baseline.py -N 25 -B 1 -T 500 -H 50 -R 30 -P 4

Adjustable parameters at runtime are
- `N` number of targets,
- `B` budget,
- `T` number of timesteps for evaluation,
- `H` number of timesteps of historical data used,
- `R` number of repeated trials, and
- `P` number of computer processes used.

Other settings in the paper are `-N 25 -B 5`, `-N 100 -B 5`, and `-N 100 -B 10`.
