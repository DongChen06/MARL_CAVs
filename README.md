# Deep Multi-agent Reinforcement Learning for Highway On-Ramp Merging in Mixed Traffic

This branch is the implementation for the MPC method proposed in [1, 2] for the traffic scenarion as shown in Fig. 1.



<p align="center">
     <img src="docs/on-ramp.png" alt="output_example" width="50%" height="50%">
     <br>Fig.1 Illustration of the considered on-ramp merging traffic scenario. CAVs (blue) and HDVs (green) coexist on both ramp and through lanes.
</p>

## Implementation
To adapt the MPC approach to the Highway environment, there are some changes we made:
- We followed [1, 2] and used a simplified point mass model for each vehicle.
- In the MPC implementation, the lane changing task is a trajectory based one, since we are modeling each vehicle as a point mass and there is no applicable "left lane" or "right lane" command. This is different from the highway simulation platform's lane changing model as in the highway simulation platform the lane changing task for a vehicle follows an exact "left lane" or "right lane" command.
- The main lane HDVs in the MPC scheme are considered to have a constant speed which is different than the IDM model considered for the HDVs in the highway simulator.

## Usage
To run the code, just run it via `python mpc_main.py`.  The config files contain the parameters for the MPC policy.


## Cite
```
@misc{chen2021deep,
      title={Deep Multi-agent Reinforcement Learning for Highway On-Ramp Merging in Mixed Traffic}, 
      author={Dong Chen and Zhaojian Li and Yongqiang Wang and Longsheng Jiang and Yue Wang},
      year={2021},
      eprint={2105.05701},
      archivePrefix={arXiv},
      primaryClass={eess.SY}
}
```

## Reference
- [1] W. Cao, M. Mukai, and T. Kawabe, “Two-dimensional merging path generation using model predictivecontrol,”Artificial Life and Robotics, vol. 17, no. 3-4, pp. 350–356, 2013
- [2]  W. Cao, M. Mukai, T. Kawabe, H. Nishira, and N. Fujiki, “Cooperative vehicle path generation duringmerging using model predictive control with real-time optimization,”Control  Engineering  Practice,vol. 34, pp. 98–105, 2015.
- [Highway-env](https://github.com/eleurent/highway-env)


