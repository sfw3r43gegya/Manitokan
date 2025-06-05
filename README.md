# The challenge of hidden gifts in multi-agent reinforcement learning
------
## Dane Malenfant, Blake Aaron Richards 2025

This repository implements Manitokan task as detailed in the paper: https://arxiv.org/pdf/2505.20579 as well as the models in the paper.

## Setup
---
All experiments used python3.9
```bash
python3.9 -m venv env_name
```

```bash
pip install -r requirements.txt
```
Another requirements file helps avoid package conflicts.

```bash
pip install -r requirements2.txt
```

All hyperparemeters can be found in the config directory in the relevant .yaml file.

To run the task:

```bash
python main.py
```

Arguments can be pass through main.py as well

```bash
python main.py --config="mappo" --env-config="manito"
```

Please reach out with any implementation issues.

## Citation
---
```bibtex
@article{malenfant2025challenge,
  title={The challenge of hidden gifts in multi-agent reinforcement learning},
  author={Malenfant, Dane and Richards, Blake A},
  journal={arXiv preprint arXiv:2505.20579},
  year={2025}
}
```
