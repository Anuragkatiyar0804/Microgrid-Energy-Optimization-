import torch
from agents.network import DQN

STATE_SIZE = 6
ACTION_SIZE = 3


network = DQN(
    state_size=STATE_SIZE,
    action_size=ACTION_SIZE
)


state = torch.tensor([
    2.5,    # load
    1.5,    # solar
    10.0,   # grid price
    0.60,   # SOC
    0.50,   # sin(hour)
    0.86    # cos(hour)
], dtype=torch.float32)


state = state.unsqueeze(0)


q_values = network(state)


print("State shape:")
print(state.shape)

print("\nQ-values:")
print(q_values)

print("\nQ-value shape:")
print(q_values.shape)

print("\nBest action:")
print(torch.argmax(q_values).item())