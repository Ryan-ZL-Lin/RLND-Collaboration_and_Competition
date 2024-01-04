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


## Replay Buffer
Reply Buffer plays an important role in MADDPG, I tested out two different modes since I'm not sure which one is better

### Mode A (the training reuslts are in checkpoints/v1)
Two agents shared the same Reply Buffer, however, during the learning process, each agent only takes its own experience from Reply Buffer and update its own Critic model weights.

### Mode B (the training results are saved in checkpoints/v2)
