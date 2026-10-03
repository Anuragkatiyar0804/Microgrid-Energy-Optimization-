from environment.microgrid_env import (
    IDLE,
    CHARGE,
    DISCHARGE
)


class RuleBasedController:
    def __init__(self, env):
        self.env = env

    def select_action(self):
        row = self.env.data.iloc[
            self.env.current_step
        ]

        load = float(row["load_kw"])
        solar = float(row["solar_kw"])

        # Solar surplus
        if solar > load:
            return CHARGE

        # Solar deficit
        if solar < load:

            if self.env.battery.can_discharge():
                return DISCHARGE

            return IDLE

        # Solar exactly matches load
        return IDLE