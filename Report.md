# Learning Algorithm
In this project, I choose [MADDPG](https://arxiv.org/pdf/1706.02275.pdf) as the algorithm to implement. Becasue I already completed the single DDPG agent previously, it's easier for me to move to MADDPG.
There are eight Deep Neural Networks in total, they are 
- Agent 1
  - Actor
      - Local
      - Target
  - Critic
      - Local
      - Target
- Agent 2
  - Actor
      - Local
      - Target
  - Critic
      - Local
      - Target
   
The corresponding model weights could be found in checkpoints/ folder.
   
## Parameters Used in Neural Network
To compare the training speed and the final outcome, I used different units in layer to train the model.
### Mode B (the training results are saved in checkpoints/v2) - 256 units in first layer and 128 units in second layer (for both Actor and Critic)
Here is the training result, it takes 920 episodes with 1 hour 8 minutes.
<img width="631" alt="training Result_v2" src="https://github.com/Ryan-ZL-Lin/RLND-Collaboration_and_Competition/assets/33056320/f6d8af91-9c33-4aec-a7e9-18055cdb033d">

### Mode C (the training results are saved in checkpoints/v3) - 128 units in frist layer and 64 units in second layer (for both Actor and Critic)
Here is the training result, it takes 1033 episodes with 1 hour and 4 minutes.
<img width="628" alt="training Result_v3" src="https://github.com/Ryan-ZL-Lin/RLND-Collaboration_and_Competition/assets/33056320/aa4a723b-b273-4cfc-a9b0-fd063dacfcfc">

**Conclusion: Smaller neural network gives faster training progress, but it doesn't mean the training can achieve the goal in lesser episodes.**

## Replay Buffer
Reply Buffer plays an important role in MADDPG, I tested out two different modes since I'm not sure which one is better

### Mode A (the training reuslts are saved in checkpoints/v1)
Two agents shared the same Reply Buffer, however, during the learning process, each agent only takes its own experience from Reply Buffer and update its own Critic model weights. It's similar to the approch used [HERE](https://github.com/JKWalleiee/Udacity-DRL-collab-compet/blob/main/maddpg_agents.py)  
The following chart shows the training result, which achieves +0.5 in 2871 episodes by taking around 3 hours.
<img width="662" alt="Training Result_v1" src="https://github.com/Ryan-ZL-Lin/RLND-Collaboration_and_Competition/assets/33056320/a0e4b47e-6e31-41d4-9427-868071a150f1">


### Mode B (the training results are saved in checkpoints/v2)
Two agents shared the same Reply Buffer, each agent just takes random exeprience from the Reply Buffer to go for ther own learning process and update their own Critic model weights. It's similar to the approach used [HERE](https://github.com/ravishchawla/Reinforcement-Learning-NanoDegree/blob/master/Project%203%20-%20Collaboration%20and%20Competition/multiagents.py)
The following chart shows the training result, which achieves +0.5 in 920 episodes by taking around 1 hour.  
<img width="631" alt="training Result_v2" src="https://github.com/Ryan-ZL-Lin/RLND-Collaboration_and_Competition/assets/33056320/88b33de3-f5ec-43d2-a2b4-fb6df3642c34">

## Hyper Parameters

