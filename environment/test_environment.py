from data.loader import load_csv
from environment.battery import Battery
from environment.microgrid_env import (
    MicrogridEnv,
    IDLE,
    CHARGE,
    DISCHARGE
)


# Load dataset
data = load_csv(
    "data/raw/microgrid.csv"
)


# Create battery
battery = Battery(
    capacity_kwh=10,
    initial_soc=0.50,
    min_soc=0.20,
    max_soc=1.00,
    charge_efficiency=0.95,
    discharge_efficiency=0.95,
    max_charge_power_kw=3,
    max_discharge_power_kw=3,
    time_step_hours=1.0
)


# Create environment
env = MicrogridEnv(
    data=data,
    battery=battery
)


# Reset
state = env.reset()

total_cost = 0

for step in range(24):

    next_state, reward, done, info = env.step(
        DISCHARGE
    )

    total_cost += info["grid_cost"]

    print(
        f"Hour {step}: "
        f"Load={info['load']:.2f}, "
        f"Solar={info['solar']:.2f}, "
        f"Battery={info['battery_discharge']:.2f}, "
        f"Grid={info['grid_used']:.2f}, "
        f"SOC={info['soc']:.2f}, "
        f"Cost={info['grid_cost']:.2f}"
    )

    state = next_state

    if done:
        break

print("\nTotal cost:", total_cost)