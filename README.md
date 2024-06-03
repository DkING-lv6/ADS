
# [ICML 2024] Improving Generalization in Offline Reinforcement Learning via Adversarial Data Splitting

This repository is based on [OfflineRL-Kit] https://github.com/yihaosun1124/OfflineRL-Kit

## Installation
First, install MuJuCo engine, which can be download from [here](https://mujoco.org/download), and install `mujoco-py` (its version depends on the version of MuJoCo engine you have installed).

Second, install D4RL:
```shell
git clone https://github.com/Farama-Foundation/d4rl.git
cd d4rl
pip install -e .
```

Finally, install OfflineRL-Kit!
```shell
git clone https://github.com/yihaosun1124/OfflineRL-Kit.git
cd OfflineRL-Kit
python setup.py install
```

## Quick Start
### Train
This is an example of CQL+ADS. You can run the full script at [run_example/run_cql_ads.py](https://github.com/yihaosun1124/OfflineRL-Kit/blob/main/run_example/run_cql.py).


```shell
python run_cql_ads.py --task "hopper-medium-v2" --seed 0 --device "cuda"
```

