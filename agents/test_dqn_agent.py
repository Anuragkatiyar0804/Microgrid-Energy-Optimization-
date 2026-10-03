import numpy as np

from agents.dqn_agent import DQNAgent


agent = DQNAgent(
    state_size=6,
    action_size=3
)


state = np.array([
    2.5,    # load
    1.5,    # solar
    10.0,   # price
    0.60,   # SOC
    0.50,   # hour sin
    0.86    # hour cos
], dtype=np.float32)


print("Device:", agent.device)
print("Initial epsilon:", agent.epsilon)


# Test action selection
action = agent.select_action(
    state
)

print("Selected action:", action)


# Add experiences
for i in range(100):

    next_state = state + np.random.normal(
        0,
        0.1,
        size=6
    ).astype(np.float32)

    action = np.random.randint(0, 3)

    reward = np.random.uniform(
        -10,
        0
    )

    done = False

    agent.remember(
        state,
        action,
        reward,
        next_state,
        done
    )


print("Replay buffer size:", len(agent.memory))


# Train
loss = agent.train_step()

print("Training loss:", loss)

print("Epsilon after training:", agent.epsilon)