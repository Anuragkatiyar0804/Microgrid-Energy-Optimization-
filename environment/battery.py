class Battery:
    def __init__(
        self,
        capacity_kwh,
        initial_soc,
        min_soc,
        max_soc,
        charge_efficiency,
        discharge_efficiency,
        max_charge_power_kw,
        max_discharge_power_kw,
        time_step_hours=1.0
    ):
        self.capacity_kwh = capacity_kwh
        self.initial_soc = initial_soc
        self.soc = initial_soc

        self.min_soc = min_soc
        self.max_soc = max_soc

        self.charge_efficiency = charge_efficiency
        self.discharge_efficiency = discharge_efficiency

        self.max_charge_power_kw = max_charge_power_kw
        self.max_discharge_power_kw = max_discharge_power_kw

        self.time_step_hours = time_step_hours

    @property
    def energy_kwh(self):
        return self.soc * self.capacity_kwh

    def reset(self):
        self.soc = self.initial_soc

    def charge(self, requested_energy_kwh):
        """
        requested_energy_kwh:
            Energy supplied to the battery before charging losses.

        Returns:
            actual_energy_input
            stored_energy
        """

        if requested_energy_kwh <= 0:
            return 0.0, 0.0

        max_power_energy = (
            self.max_charge_power_kw
            * self.time_step_hours
        )

        available_capacity = (
            self.max_soc * self.capacity_kwh
            - self.energy_kwh
        )

        # Because of efficiency, we need to account
        # for the energy that will actually be stored.
        max_input_for_capacity = (
            available_capacity / self.charge_efficiency
        )

        actual_input = min(
            requested_energy_kwh,
            max_power_energy,
            max_input_for_capacity
        )

        stored_energy = (
            actual_input * self.charge_efficiency
        )

        self.soc += stored_energy / self.capacity_kwh

        self.soc = min(self.soc, self.max_soc)

        return actual_input, stored_energy

    def discharge(self, requested_energy_kwh):
        """
        requested_energy_kwh:
            Energy requested from the battery output.

        Returns:
            actual_energy_output
            energy_removed_from_battery
        """

        if requested_energy_kwh <= 0:
            return 0.0, 0.0

        max_power_energy = (
            self.max_discharge_power_kw
            * self.time_step_hours
        )

        available_energy = (
            self.energy_kwh
            - self.min_soc * self.capacity_kwh
        )

        # Due to discharge efficiency, more energy
        # must be removed from the battery than reaches load.
        max_output_from_soc = (
            available_energy * self.discharge_efficiency
        )

        actual_output = min(
            requested_energy_kwh,
            max_power_energy,
            max_output_from_soc
        )

        energy_removed = (
            actual_output / self.discharge_efficiency
        )

        self.soc -= energy_removed / self.capacity_kwh

        self.soc = max(self.soc, self.min_soc)

        return actual_output, energy_removed

    def can_charge(self):
        return self.soc < self.max_soc

    def can_discharge(self):
        return self.soc > self.min_soc