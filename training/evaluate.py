import numpy as np

from data.loader import load_csv
from environment.battery import Battery
from environment.microgrid_env import MicrogridEnv
from baselines.rule_based import RuleBasedController
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

NUM_TEST_DAYS = 73


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


def evaluate_dqn(data):
    battery = create_battery()

    env = MicrogridEnv(
        data=data,
        battery=battery,
        episode_length=24
    )

    agent = create_agent()
    agent.load(MODEL_PATH)

    # No exploration during evaluation
    agent.epsilon = 0.0

    results = []

    # First 30 complete days
    for day in range(NUM_TEST_DAYS):
        start_step = day * 24

        state = env.reset(start_step=start_step)

        total_cost = 0.0
        total_grid = 0.0
        total_solar = 0.0
        total_solar_used = 0.0
        total_curtailed = 0.0
        total_charge = 0.0
        total_discharge = 0.0
        soc_sum = 0.0

        done = False

        while not done:
            action = agent.select_action(
                state,
                training=False
            )

            next_state, reward, done, info = env.step(action)

            total_cost += info["grid_cost"]
            total_grid += info["grid_used"]
            total_solar += info["solar"]
            total_solar_used += info["solar_used"]
            total_curtailed += info["curtailed_solar"]
            total_charge += info["battery_charge"]
            total_discharge += info["battery_discharge"]
            soc_sum += info["soc"]

            state = next_state

        results.append({
            "day": day + 1,
            "cost": total_cost,
            "grid_energy": total_grid,
            "solar_energy": total_solar,
            "solar_used": total_solar_used,
            "curtailed_solar": total_curtailed,
            "battery_charge": total_charge,
            "battery_discharge": total_discharge,
            "average_soc": soc_sum / 24
        })

    return results


def evaluate_rule_based(data):
    battery = create_battery()

    env = MicrogridEnv(
        data=data,
        battery=battery,
        episode_length=24
    )

    controller = RuleBasedController(env)

    results = []

    for day in range(NUM_TEST_DAYS):
        start_step = day * 24

        state = env.reset(start_step=start_step)

        total_cost = 0.0
        total_grid = 0.0
        total_solar = 0.0
        total_solar_used = 0.0
        total_curtailed = 0.0
        total_charge = 0.0
        total_discharge = 0.0
        soc_sum = 0.0

        done = False

        while not done:
            action = controller.select_action()

            next_state, reward, done, info = env.step(action)

            total_cost += info["grid_cost"]
            total_grid += info["grid_used"]
            total_solar += info["solar"]
            total_solar_used += info["solar_used"]
            total_curtailed += info["curtailed_solar"]
            total_charge += info["battery_charge"]
            total_discharge += info["battery_discharge"]
            soc_sum += info["soc"]

            state = next_state

        results.append({
            "day": day + 1,
            "cost": total_cost,
            "grid_energy": total_grid,
            "solar_energy": total_solar,
            "solar_used": total_solar_used,
            "curtailed_solar": total_curtailed,
            "battery_charge": total_charge,
            "battery_discharge": total_discharge,
            "average_soc": soc_sum / 24
        })

    return results


def summarize(name, results):
    costs = np.array([r["cost"] for r in results])
    grid = np.array([r["grid_energy"] for r in results])
    solar = np.array([r["solar_energy"] for r in results])
    solar_used = np.array([r["solar_used"] for r in results])
    curtailed = np.array([r["curtailed_solar"] for r in results])
    charge = np.array([r["battery_charge"] for r in results])
    discharge = np.array([r["battery_discharge"] for r in results])
    soc = np.array([r["average_soc"] for r in results])

    solar_utilization = (
        solar_used.sum() / solar.sum() * 100
        if solar.sum() > 0
        else 0
    )

    return {
        "name": name,
        "total_cost": costs.sum(),
        "average_daily_cost": costs.mean(),
        "total_grid_energy": grid.sum(),
        "average_daily_grid_energy": grid.mean(),
        "solar_utilization": solar_utilization,
        "curtailed_solar": curtailed.sum(),
        "battery_charge": charge.sum(),
        "battery_discharge": discharge.sum(),
        "average_soc": soc.mean()
    }


def print_comparison(rule_summary, dqn_summary):
    print("\n" + "=" * 60)
    print("DQN VS RULE-BASED COMPARISON")
    print("=" * 60)

    print(f"\nTest days: {NUM_TEST_DAYS}")

    print("\nMetric                    Rule-Based       DQN")
    print("-" * 60)

    print(
        f"Total Cost               ₹{rule_summary['total_cost']:>10.2f}"
        f"       ₹{dqn_summary['total_cost']:>10.2f}"
    )

    print(
        f"Average Daily Cost       ₹{rule_summary['average_daily_cost']:>10.2f}"
        f"       ₹{dqn_summary['average_daily_cost']:>10.2f}"
    )

    print(
        f"Total Grid Energy        {rule_summary['total_grid_energy']:>10.2f}"
        f"       {dqn_summary['total_grid_energy']:>10.2f} kWh"
    )

    print(
        f"Solar Utilization        {rule_summary['solar_utilization']:>10.2f}%"
        f"       {dqn_summary['solar_utilization']:>10.2f}%"
    )

    print(
        f"Curtailed Solar          {rule_summary['curtailed_solar']:>10.2f}"
        f"       {dqn_summary['curtailed_solar']:>10.2f} kWh"
    )

    print(
        f"Battery Charge           {rule_summary['battery_charge']:>10.2f}"
        f"       {dqn_summary['battery_charge']:>10.2f} kWh"
    )

    print(
        f"Battery Discharge        {rule_summary['battery_discharge']:>10.2f}"
        f"       {dqn_summary['battery_discharge']:>10.2f} kWh"
    )

    print(
        f"Average SOC              {rule_summary['average_soc']:>10.2f}"
        f"       {dqn_summary['average_soc']:>10.2f}"
    )

    cost_improvement = (
        (rule_summary["total_cost"] - dqn_summary["total_cost"])
        / rule_summary["total_cost"]
        * 100
    )

    print("\n" + "-" * 60)
    print(f"DQN Cost Improvement: {cost_improvement:.2f}%")
    print("=" * 60)


def main():
    print("Loading dataset...")

    data = load_csv(DATA_PATH)
    split_index = (int(len(data) * 0.80) // 24) * 24
    test_data = data.iloc[split_index:].reset_index(drop=True)
   
    print(f"Dataset loaded: {len(test_data)} rows")

    print("\nEvaluating Rule-Based controller...")
    rule_results = evaluate_rule_based(test_data)

    print("Evaluating DQN...")
    dqn_results = evaluate_dqn(test_data)

    rule_summary = summarize(
        "Rule-Based",
        rule_results
    )

    dqn_summary = summarize(
        "DQN",
        dqn_results
    )

    print_comparison(
        rule_summary,
        dqn_summary
    )


if __name__ == "__main__":
    main()