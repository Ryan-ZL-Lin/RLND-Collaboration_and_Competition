# Default hyperparameters
                           
BUFFER_SIZE = int(1e5)             # replay buffer size
BATCH_SIZE = 1024                  # minibatch size

FA1_UNITS = 128  #256              # Number of units for the layer 1 in the actor model
FA2_UNITS = 64  #128              # Number of units for the layer 2 in the actor model
FC1_UNITS = 128  #256              # Number of units for the layer 1 in the critic model
FC2_UNITS = 64  #128              # Number of units for the layer 2 in the critic model
LR_ACTOR = 1e-3    #1e-4           # learning rate of the actor 
LR_CRITIC = 1e-3   #2e-3           # learning rate of the critic
WEIGHT_DECAY = 0   #0.0001         # L2 weight decay

GAMMA = 0.99 #0.995                # Discount factor
TAU = 1e-3                         # For soft update of target parameters

MU = 0.                            # Ornstein-Uhlenbeck noise parameter
THETA = 0.15                       # Ornstein-Uhlenbeck noise parameter
SIGMA = 0.1                        # Ornstein-Uhlenbeck noise parameter
NOISE = 1.0                        # Initial Noise Amplitude 
NOISE_REDUCTION = 1.0 # 0.995      # Noise amplitude decay ratio