# LeRobot

## Installation

Install dependencies using `conda`:

```
conda env create -f environment.yaml
conda activate lerobot
```

**dev**

```
python setup.py develop
```

## TODO

- [x] priority update doesnt match FOWM or original paper
- [x] self.step=100000 should be updated at every step to adjust to horizon of planner
- [ ] prefetch replay buffer to speedup training
- [ ] parallelize env to speedup eval

## Contribute

**style**
```
isort lerobot
black lerobot
isort test
black test
pylint lerobot
```
