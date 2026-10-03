from data.loader import load_csv
from environment.battery import Battery
from environment.microgrid_env import MicrogridEnv
from baselines.rule_based import RuleBasedController


# Load data
data = load_csv(
    "data/raw/microgrid.csv"
)


# Battery
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


# Environment
env = MicrogridEnv(
    data=data,
    battery=battery
)


controller = RuleBasedController(env)

state = env.reset()

total_cost = 0.0

for step in range(24):

    action = controller.select_action()

    next_state, reward, done, info = env.step(
        action
    )

    total_cost += info["grid_cost"]

    print(
        f"Hour {step:02d} | "
        f"Load={info['load']:.2f} | "
        f"Solar={info['solar']:.2f} | "
        f"Battery Charge={info['battery_charge']:.2f} | "
        f"Battery Discharge={info['battery_discharge']:.2f} | "
        f"Grid={info['grid_used']:.2f} | "
        f"SOC={info['soc']:.2f} | "
        f"Cost=₹{info['grid_cost']:.2f}"
    )

    state = next_state

    if done:
        break


print("\nTotal cost for 24 hours:")
print(f"₹{total_cost:.2f}")