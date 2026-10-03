import numpy as np

from agents.replay_buffer import ReplayBuffer


buffer = ReplayBuffer(
    capacity=5
)


# Add some fake experiences
for i in range(5):

    state = np.array([
        i,
        i + 1,
        i + 2,
        0.5,
        0.0,
        1.0
    ], dtype=np.float32)

    action = i % 3

    reward = -float(i)

    next_state = state + 0.1

    done = False

    buffer.push(
        state,
        action,
        reward,
        next_state,
        done
    )


print("Buffer size:", len(buffer))


# Sample experiences
batch = buffer.sample(3)

print("\nSampled experiences:")

for experience in batch:
    state, action, reward, next_state, done = experience

    print(
        "Action:",
        action,
        "Reward:",
        reward,
        "Done:",
        done
    )