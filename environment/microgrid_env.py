import numpy as np

IDLE = 0
CHARGE = 1
DISCHARGE = 2


class MicrogridEnv:
    def __init__(self, data, battery, episode_length=24):
        self.data = data.reset_index(drop=True)
        self.battery = battery
        self.episode_length = episode_length
        self.current_step = 0
        self.start_step = 0
        self.end_step = 0

    def reset(self, start_step=None):
        self.battery.reset()

        if start_step is None:
            max_start = len(self.data) - self.episode_length
            self.start_step = np.random.randint(0, max_start + 1)
        else:
            self.start_step = start_step

        self.current_step = self.start_step
        self.end_step = self.start_step + self.episode_length

        return self._get_state()

    def _get_state(self):
        row = self.data.iloc[self.current_step]

        load = float(row["load_kw"])
        solar = float(row["solar_kw"])
        price = float(row["grid_price"])

        hour = row["timestamp"].hour

        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)

        return np.array([
            load,
            solar,
            price,
            self.battery.soc,
            hour_sin,
            hour_cos
        ], dtype=np.float32)

    def step(self, action):
        row = self.data.iloc[self.current_step]

        load = float(row["load_kw"])
        solar = float(row["solar_kw"])
        grid_price = float(row["grid_price"])

        # --------------------------------------------------
        # 1. Solar serves the load first
        # --------------------------------------------------
        solar_used = min(load, solar)

        remaining_load = load - solar_used
        solar_surplus = max(solar - load, 0.0)

        battery_charge = 0.0
        battery_discharge = 0.0
        grid_used = 0.0
        curtailed_solar = 0.0

        # --------------------------------------------------
        # 2. Agent decides battery action
        # --------------------------------------------------

        if action == CHARGE:
            # Charge only from excess solar
            if solar_surplus > 0:
                battery_charge, _ = self.battery.charge(solar_surplus)

                curtailed_solar = (
                    solar_surplus - battery_charge
                )
            else:
                # No excess solar -> nothing to charge
                grid_used = remaining_load

        elif action == DISCHARGE:
            # Battery helps serve remaining load
            if remaining_load > 0:
                battery_discharge, _ = self.battery.discharge(
                    remaining_load
                )

                remaining_load -= battery_discharge

            grid_used = remaining_load

            # Any solar surplus is curtailed.
            # The agent explicitly chose discharge.
            curtailed_solar = solar_surplus

        elif action == IDLE:
            # Do not charge or discharge battery
            grid_used = remaining_load
            curtailed_solar = solar_surplus

        else:
            raise ValueError(f"Invalid action: {action}")

        # --------------------------------------------------
        # 3. Calculate cost
        # --------------------------------------------------

        grid_cost = grid_used * grid_price

        battery_energy_moved = (
            battery_charge + battery_discharge
        )

        # Main objective: minimize grid cost
        reward = -grid_cost

        # Small penalty for wasting solar
        reward -= 0.05 * curtailed_solar

        # Small battery degradation penalty
        reward -= 0.02 * battery_energy_moved

        # --------------------------------------------------
        # 4. Move to next timestep
        # --------------------------------------------------

        self.current_step += 1

        done = self.current_step >= self.end_step

        if done:
            next_state = np.zeros(6, dtype=np.float32)
        else:
            next_state = self._get_state()

        info = {
            "load": load,
            "solar": solar,
            "solar_used": solar_used,
            "solar_surplus": solar_surplus,
            "battery_charge": battery_charge,
            "battery_discharge": battery_discharge,
            "grid_used": grid_used,
            "grid_price": grid_price,
            "grid_cost": grid_cost,
            "curtailed_solar": curtailed_solar,
            "soc": self.battery.soc
        }

        return next_state, reward, done, info