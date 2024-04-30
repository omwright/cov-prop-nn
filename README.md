# An Analytic Solution to Covariance Propagation in Neural Networks

This code accompanies "[An Analytic Solution to Covariance Propagation in Neural Networks](https://arxiv.org/abs/2403.16163)" by Oren Wright, Yorie Nakahira, and José M. F. Moura.

## Dependencies

We use Python 3.10 and PyTorch 1.13.1 in our experiments, along with a variety of other packages (numpy, matplotlib, pandas, requests, torchvision, tqdm). For managing Python projects, [Pipenv](https://pipenv-fork.readthedocs.io/en/latest/) or [Anaconda](https://docs.anaconda.com/free/anaconda/) can be useful.

## Use

The `network-moments` directory contains moment propagation [code](https://github.com/xmodar/network_moments/) used in [Bibi, et al., 2018](https://ieeexplore.ieee.org/document/8579046). The `dvi` directory contains a PyTorch implementation of [deterministic variational inference](https://github.com/microsoft/deterministic-variational-inference) used in [Wu, et al., 2019](https://arxiv.org/abs/1810.03958). (PyTorch BNN implementations by [Hoki Kim](https://github.com/Harry24k/bayesian-neural-network-pytorch/) and [Alexander Markov](https://github.com/markovalexander/DVI/) were useful references in design.)

Run top-level scripts to execute the analysis and synthesis experiments in the paper. First-time executions may take longer to run due to downloading required datasets.

```bash
python run_tightness_synthetic.py
python run_tightness_mnist.py
python run_dvi.py config/your_config_here.json
```

## Cite

```bibtex
@inproceedings{wright2024analytic,
  title={An Analytic Solution to Covariance Propagation in Neural Networks},
  author={Wright, Oren and Nakahira, Yorie and Moura, Jos{\'e} MF},
  booktitle={International Conference on Artificial Intelligence and Statistics},
  pages={4087--4095},
  year={2024},
  organization={PMLR}
}
```
