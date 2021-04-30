# Deep Multi-agent Reinforcement Learning for Highway On-Ramp Merging in Mixed Traffic

An on-policy MARL algorithm for on-ramp merging, which features parameter sharing, action masking, local reward design and a priority-based safety supervisor.

## Installation
- create an python virtual environment: `conda create -n marl_cav python=3.6`
- active the virtul environment: `conda activate marl_cav`
- install pytorch (torch>=1.2.0): `pip install torch===1.7.0 torchvision===0.8.1 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html`
- install the requirements: `pip install -r requirements.txt`

<p align="center">
     <img src="docs/on-ramp.png" alt="output_example" width="80%" height="80%">
     <br>Fig.1 Illustration of the considered on-ramp merging traffic scenario. CAVs (blue) and HDVs (green) coexist on both ramp and through lanes.
</p>

## Usage
To run the code, just run it via `python run_xxx.py`.  The config files contain the parameters for the MARL policies.

## Cite
```
///
```

