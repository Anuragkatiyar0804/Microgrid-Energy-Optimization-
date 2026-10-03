import os

import numpy as np

from data.loader import load_csv
from environment.battery import Battery
from environment.microgrid_env import MicrogridEnv
from agents.dqn_agent import DQNAgent

from config.config import (
    BATTERY_CAPACITY_KWH,
    INITIAL_SOC,
    MIN_SOC,
    MAX_SOC,
    MAX_CHARGE_POWER_KW,
    MAX_DISCHARGE_POWER_KW,
    CHARGE_EFFICIENCY,
    DISCHARGE_EFFICIENCY,
    TIME_STEP_HOURS,
    STATE_SIZE,
    ACTION_SIZE,
    GAMMA,
    LEARNING_RATE,
    BATCH_SIZE,
    REPLAY_MEMORY_SIZE,
    EPSILON_START,
    EPSILON_MIN,
    EPSILON_DECAY,
    TARGET_UPDATE_FREQUENCY,
    NUM_EPISODES
)


def create_battery():

    return Battery(
        capacity_kwh=BATTERY_CAPACITY_KWH,
        initial_soc=INITIAL_SOC,
        min_soc=MIN_SOC,
        max_soc=MAX_SOC,
        charge_efficiency=CHARGE_EFFICIENCY,
        discharge_efficiency=DISCHARGE_EFFICIENCY,
        max_charge_power_kw=MAX_CHARGE_POWER_KW,
        max_discharge_power_kw=MAX_DISCHARGE_POWER_KW,
        time_step_hours=TIME_STEP_HOURS
    )


def main():

    # -----------------------------
    # Load dataset
    # -----------------------------

    data = load_csv(
        "data/raw/microgrid.csv"
    )

    print(
        f"Dataset loaded: {len(data)} rows"
    )

    split_index = (int(len(data) * 0.80) // 24) * 24

    train_data = data.iloc[:split_index].reset_index(drop=True)
    test_data = data.iloc[split_index:].reset_index(drop=True)

    print(f"Total rows: {len(data)}")
    print(f"Training rows: {len(train_data)}")
    print(f"Testing rows: {len(test_data)}")

    # -----------------------------
    # Create environment
    # -----------------------------

    battery = create_battery()

    env = MicrogridEnv(
        data=train_data,
        battery=battery,
        episode_length=24
    )

    # -----------------------------
    # Create agent
    # -----------------------------

    agent = DQNAgent(
        state_size=STATE_SIZE,
        action_size=ACTION_SIZE,
        gamma=GAMMA,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        replay_memory_size=REPLAY_MEMORY_SIZE,
        epsilon_start=EPSILON_START,
        epsilon_min=EPSILON_MIN,
        epsilon_decay=EPSILON_DECAY,
        target_update_frequency=TARGET_UPDATE_FREQUENCY
    )

    # -----------------------------
    # Training
    # -----------------------------

    episode_rewards = []

    for episode in range(
        1,
        NUM_EPISODES + 1
    ):

        state = env.reset()

        total_reward = 0.0
        total_cost = 0.0

        done = False

        while not done:

            action = agent.select_action(
                state,
                training=True
            )

            (
                next_state,
                reward,
                done,
                info
            ) = env.step(action)

            agent.remember(
                state,
                action,
                reward,
                next_state,
                done
            )

            loss = agent.train_step()

            state = next_state

            total_reward += reward
            total_cost += info["grid_cost"]

        # Update target network
        if (
            episode
            % TARGET_UPDATE_FREQUENCY
            == 0
        ):
            agent.update_target_network()

        episode_rewards.append(
            total_reward
        )

        # Print progress
        if episode % 10 == 0:

            avg_reward = np.mean(
                episode_rewards[-10:]
            )

            print(
                f"Episode {episode:4d} | "
                f"Reward: {total_reward:8.2f} | "
                f"Avg Reward: {avg_reward:8.2f} | "
                f"Cost: ₹{total_cost:7.2f} | "
                f"Epsilon: {agent.epsilon:.4f}"
            )

    # -----------------------------
    # Save model
    # -----------------------------

    os.makedirs(
        "models",
        exist_ok=True
    )

    model_path = (
        "models/dqn_microgrid.pth"
    )

    agent.save(model_path)

    print(
        f"\nModel saved to: {model_path}"
    )


if __name__ == "__main__":
    main()