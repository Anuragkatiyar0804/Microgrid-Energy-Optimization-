from data.loader import load_csv
from environment.battery import Battery
from environment.microgrid_env import MicrogridEnv, IDLE, CHARGE, DISCHARGE
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
    EPSILON_DECAY
)

DATA_PATH = "data/raw/microgrid.csv"
MODEL_PATH = "models/dqn_microgrid.pth"

ACTION_NAMES = {
    IDLE: "IDLE",
    CHARGE: "CHARGE",
    DISCHARGE: "DISCHARGE"
}


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


def create_agent():
    return DQNAgent(
        state_size=STATE_SIZE,
        action_size=ACTION_SIZE,
        gamma=GAMMA,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        replay_memory_size=REPLAY_MEMORY_SIZE,
        epsilon_start=EPSILON_START,
        epsilon_min=EPSILON_MIN,
        epsilon_decay=EPSILON_DECAY
    )


def main():
    data = load_csv(DATA_PATH)

    battery = create_battery()

    env = MicrogridEnv(
        data=data,
        battery=battery,
        episode_length=24
    )

    agent = create_agent()
    agent.load(MODEL_PATH)
    agent.epsilon = 0.0

    # Analyze the first complete day
    state = env.reset(start_step=0)

    total_cost = 0.0
    total_grid = 0.0
    total_charge = 0.0
    total_discharge = 0.0

    print("\nDQN ONE-DAY ANALYSIS")
    print("=" * 105)

    print(
        f"{'Hour':>4} "
        f"{'Load':>7} "
        f"{'Solar':>7} "
        f"{'Price':>7} "
        f"{'SOC':>7} "
        f"{'Action':>11} "
        f"{'Grid':>7} "
        f"{'Charge':>8} "
        f"{'Discharge':>10} "
        f"{'Cost':>8}"
    )

    print("-" * 105)

    done = False

    while not done:
        hour = env.data.iloc[env.current_step]["timestamp"].hour

        action = agent.select_action(
            state,
            training=False
        )

        next_state, reward, done, info = env.step(action)

        action_name = ACTION_NAMES[action]

        print(
            f"{hour:>4} "
            f"{info['load']:>7.2f} "
            f"{info['solar']:>7.2f} "
            f"{info['grid_price']:>7.2f} "
            f"{info['soc']:>7.2f} "
            f"{action_name:>11} "
            f"{info['grid_used']:>7.2f} "
            f"{info['battery_charge']:>8.2f} "
            f"{info['battery_discharge']:>10.2f} "
            f"{info['grid_cost']:>8.2f}"
        )

        total_cost += info["grid_cost"]
        total_grid += info["grid_used"]
        total_charge += info["battery_charge"]
        total_discharge += info["battery_discharge"]

        state = next_state

    print("-" * 105)

    print(f"Total Grid Cost:        ₹{total_cost:.2f}")
    print(f"Total Grid Energy:      {total_grid:.2f} kWh")
    print(f"Total Battery Charge:   {total_charge:.2f} kWh")
    print(f"Total Battery Discharge:{total_discharge:.2f} kWh")
    print(f"Final Battery SOC:      {battery.soc:.2f}")
    print("=" * 105)


if __name__ == "__main__":
    main()