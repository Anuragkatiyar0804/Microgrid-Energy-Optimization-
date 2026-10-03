from battery import Battery


battery = Battery(
    capacity_kwh=10,
    initial_soc=0.50,
    min_soc=0.20,
    max_soc=1.00,
    charge_efficiency=0.95,
    discharge_efficiency=0.95,
    max_charge_power_kw=3,
    max_discharge_power_kw=3
)

print("Initial SOC:", battery.soc)
print("Initial energy:", battery.energy_kwh)


# Charge 2 kWh
actual_input, stored = battery.charge(2)

print("\n--- Charge ---")
print("Energy requested:", 2)
print("Energy accepted:", actual_input)
print("Energy stored:", stored)
print("SOC:", battery.soc)


# Discharge 1 kWh
actual_output, removed = battery.discharge(1)

print("\n--- Discharge ---")
print("Energy requested:", 1)
print("Energy delivered:", actual_output)
print("Energy removed:", removed)
print("SOC:", battery.soc)