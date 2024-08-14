import pandas as pd
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus
import numpy as np
import matplotlib.pyplot as plt

# Load the demand profile and generation profiles from the Excel files
demand_profile = pd.read_excel('demand_profile.xlsx', header=0)
generation_profiles = pd.read_excel('generation_profiles.xlsx', header=0)

# Define capacities for wind and solar (in MW)
capacity_wind = 25
capacity_solar = 25

# Calculate the actual generation profiles by multiplying capacity factors by the defined capacities
solar_generation = generation_profiles['Solar Capacity Factor'] * capacity_solar
wind_generation = generation_profiles['Wind Capacity Factor'] * capacity_wind

# Extract the demand profile
demand = demand_profile['Demand']

# Ensure the data is in the correct format (168 hourly intervals)
data = pd.DataFrame({
    'Solar Generation (MW)': solar_generation,
    'Wind Generation (MW)': wind_generation,
    'Demand (MW)': demand
})

# Storage efficiencies and capacities
efficiency_ldes = 0.85
efficiency_sdes = 0.9
efficiency_hydrogen = 0.75

capacity_ldes = 1000  # in MWh
capacity_sdes = 500   # in MWh
capacity_hydrogen = 1500  # in MWh

# Costs (example values)
cost_solar = 10  # £/MWh
cost_wind = 15  # £/MWh
cost_ldes_charge = 5  # £/MWh
cost_ldes_discharge = 7  # £/MWh
cost_sdes_charge = 3  # £/MWh
cost_sdes_discharge = 5  # £/MWh
cost_hydrogen_charge = 8  # £/MWh
cost_hydrogen_discharge = 10  # £/MWh
cost_unmet_demand = 100  # £/MWh for unmet demand (high penalty)
cost_curtailment = 5

# Create the LP problem
prob = LpProblem("Least_Cost_Dispatch", LpMinimize)

# Define decision variables
Charge_LDES = LpVariable.dicts("Charge_LDES", range(168), lowBound=0, cat='Continuous')
Discharge_LDES = LpVariable.dicts("Discharge_LDES", range(168), lowBound=0, cat='Continuous')

Charge_SDES = LpVariable.dicts("Charge_SDES", range(168), lowBound=0, cat='Continuous')
Discharge_SDES = LpVariable.dicts("Discharge_SDES", range(168), lowBound=0, cat='Continuous')

Charge_Hydrogen = LpVariable.dicts("Charge_Hydrogen", range(168), lowBound=0, cat='Continuous')
Discharge_Hydrogen = LpVariable.dicts("Discharge_Hydrogen", range(168), lowBound=0, cat='Continuous')

Unmet_Demand = LpVariable.dicts("Unmet_Demand", range(168), lowBound=0, cat='Continuous')
Curtailment = LpVariable.dicts("Curtailment", range(168), lowBound=0, cat='Continuous')

# Objective: Minimize the total cost of generation, charging, discharging, and unmet demand
prob += lpSum([
    cost_solar * data['Solar Generation (MW)'][t] +
    cost_wind * data['Wind Generation (MW)'][t] +
    cost_ldes_charge * Charge_LDES[t] +
    cost_ldes_discharge * Discharge_LDES[t] +
    cost_sdes_charge * Charge_SDES[t] +
    cost_sdes_discharge * Discharge_SDES[t] +
    cost_hydrogen_charge * Charge_Hydrogen[t] +
    cost_hydrogen_discharge * Discharge_Hydrogen[t] +
    cost_unmet_demand * Unmet_Demand[t] +
    cost_curtailment * Curtailment[t]  # Assuming zero cost for curtailment
    for t in range(168)
]), "Total_Cost"

# Set the initial state of charge (SOC) for each storage type to zero
SOC_LDES = {0: 0}
SOC_SDES = {0: 0}
SOC_Hydrogen = {0: 0}

# Additional constraint to ensure storage discharge only happens to meet demand
for t in range(1, 168):
    # Define the SOC for each storage at time t
    SOC_LDES[t] = LpVariable(f"SOC_LDES_{t}", lowBound=0, upBound=capacity_ldes)
    SOC_SDES[t] = LpVariable(f"SOC_SDES_{t}", lowBound=0, upBound=capacity_sdes)
    SOC_Hydrogen[t] = LpVariable(f"SOC_Hydrogen_{t}", lowBound=0, upBound=capacity_hydrogen)

    # Storage dynamics
    prob += SOC_LDES[t] == SOC_LDES[t-1] + Charge_LDES[t] * efficiency_ldes - Discharge_LDES[t] * (1 / efficiency_ldes), f"LDES_SOC_{t}"
    prob += SOC_SDES[t] == SOC_SDES[t-1] + Charge_SDES[t] * efficiency_sdes - Discharge_SDES[t] * (1 / efficiency_sdes), f"SDES_SOC_{t}"
    prob += SOC_Hydrogen[t] == SOC_Hydrogen[t-1] + Charge_Hydrogen[t] * efficiency_hydrogen - Discharge_Hydrogen[t] * (1 / efficiency_hydrogen), f"Hydrogen_SOC_{t}"

    # Ensure that the discharge from storage does not exceed the available SOC at that time
    prob += Discharge_LDES[t] <= SOC_LDES[t-1], f"LDES_Discharge_Limit_{t}"
    prob += Discharge_SDES[t] <= SOC_SDES[t-1], f"SDES_Discharge_Limit_{t}"
    prob += Discharge_Hydrogen[t] <= SOC_Hydrogen[t-1], f"Hydrogen_Discharge_Limit_{t}"

    # Discharge can only happen to meet demand
    prob += Discharge_LDES[t] <= data['Demand (MW)'][t], f"LDES_Discharge_Meets_Demand_{t}"
    prob += Discharge_SDES[t] <= data['Demand (MW)'][t], f"SDES_Discharge_Meets_Demand_{t}"
    prob += Discharge_Hydrogen[t] <= data['Demand (MW)'][t], f"Hydrogen_Discharge_Meets_Demand_{t}"

    # Capacity constraints
    prob += SOC_LDES[t] <= capacity_ldes, f"LDES_Capacity_{t}"
    prob += SOC_SDES[t] <= capacity_sdes, f"SDES_Capacity_{t}"
    prob += SOC_Hydrogen[t] <= capacity_hydrogen, f"Hydrogen_Capacity_{t}"

    # Demand-Supply Constraint with Unmet Demand and Curtailed Energy
    prob += (
        data['Solar Generation (MW)'][t] +
        data['Wind Generation (MW)'][t] +
        Discharge_LDES[t] +
        Discharge_SDES[t] +
        Discharge_Hydrogen[t] +
        Unmet_Demand[t]
        == data['Demand (MW)'][t] +
        Charge_LDES[t] +
        Charge_SDES[t] +
        Charge_Hydrogen[t] +
        Curtailment[t]  # Add curtailment to ensure excess generation is handled
    ), f"Demand_Supply_Constraint_{t}"


# Final state of charge must be within capacity limits
prob += SOC_LDES[167] <= capacity_ldes, "Final_LDES_SOC"
prob += SOC_SDES[167] <= capacity_sdes, "Final_SDES_SOC"
prob += SOC_Hydrogen[167] <= capacity_hydrogen, "Final_Hydrogen_SOC"

# Solve the optimization problem
prob.solve()
print("Status:", LpStatus[prob.status])

# Display how each hour of demand is met, including all relevant components
for t in range(168):
    print(f"Hour {t}:")
    print(f"  Solar Generation: {data['Solar Generation (MW)'][t]:.2f} MW")
    print(f"  Wind Generation: {data['Wind Generation (MW)'][t]:.2f} MW")
    print(f"  LDES Discharge: {Discharge_LDES[t].varValue:.2f} MW")
    print(f"  SDES Discharge: {Discharge_SDES[t].varValue:.2f} MW")
    print(f"  Hydrogen Discharge: {Discharge_Hydrogen[t].varValue:.2f} MW")
    print(f"  Unmet Demand: {Unmet_Demand[t].varValue:.2f} MW")
    print(f"  Curtailment: {Curtailment[t].varValue:.2f} MW")
    print(f"  LDES Charge: {Charge_LDES[t].varValue:.2f} MW")
    print(f"  SDES Charge: {Charge_SDES[t].varValue:.2f} MW")
    print(f"  Hydrogen Charge: {Charge_Hydrogen[t].varValue:.2f} MW")
    print(f"  Total Demand: {data['Demand (MW)'][t]:.2f} MW")
    total_generation = (data['Solar Generation (MW)'][t] + data['Wind Generation (MW)'][t])
    total_discharge = (Discharge_LDES[t].varValue + Discharge_SDES[t].varValue + Discharge_Hydrogen[t].varValue)
    total_charge = (Charge_LDES[t].varValue + Charge_SDES[t].varValue + Charge_Hydrogen[t].varValue)
    total_curtailment = Curtailment[t].varValue
    total_unmet = Unmet_Demand[t].varValue
    balance_check = total_generation + total_discharge - total_charge - total_curtailment + total_unmet
    print(f"  Balance Check (Generation + Discharge - Charge - Curtailment - Unmet): {balance_check:.2f} MW\n")

# Prepare data for plotting
results = {
    'Solar Generation': [data['Solar Generation (MW)'][t] for t in range(168)],
    'Wind Generation': [data['Wind Generation (MW)'][t] for t in range(168)],
    'LDES Discharge': [Discharge_LDES[t].varValue for t in range(168)],
    'SDES Discharge': [Discharge_SDES[t].varValue for t in range(168)],
    'Hydrogen Discharge': [Discharge_Hydrogen[t].varValue for t in range(168)],
    'Unmet Demand': [Unmet_Demand[t].varValue for t in range(168)]
}

df_results = pd.DataFrame(results)

# Plot the stacked bar chart
fig, ax = plt.subplots(figsize=(14, 8))

# Create a stacked bar chart
df_results.plot(kind='bar', stacked=True, ax=ax, width=1)

# Plot the demand line
ax.plot(df_results.index, data['Demand (MW)'], label='Demand', linestyle='--', color='black', linewidth=2)

# Customize the plot
ax.set_title('How Demand is Met at Each Interval')
ax.set_xlabel('Hour')
ax.set_ylabel('Power (MW)')
ax.legend(loc='upper right')
ax.grid(True, axis='y')

plt.tight_layout()

# Save the figure to a file
plt.savefig('demand_vs_supply.png')

# Optionally, you can also display the plot if running in a notebook
plt.show()

# Prepare data for SOC plotting
soc_results = {
    'LDES SOC': [SOC_LDES[t].varValue for t in range(1, 168)],
    'SDES SOC': [SOC_SDES[t].varValue for t in range(1, 168)],
    'Hydrogen SOC': [SOC_Hydrogen[t].varValue for t in range(1, 168)]
}

df_soc_results = pd.DataFrame(soc_results)

# Plot the SOCs on the same graph
plt.figure(figsize=(14, 8))

plt.plot(df_soc_results.index, df_soc_results['LDES SOC'], label='LDES SOC', color='blue', linewidth=2)
plt.plot(df_soc_results.index, df_soc_results['SDES SOC'], label='SDES SOC', color='green', linewidth=2)
plt.plot(df_soc_results.index, df_soc_results['Hydrogen SOC'], label='Hydrogen SOC', color='red', linewidth=2)

# Customize the plot
plt.title('State of Charge (SOC) for All Storage Systems')
plt.xlabel('Hour')
plt.ylabel('State of Charge (MWh)')
plt.legend(loc='upper right')
plt.grid(True, axis='y')

plt.tight_layout()

# Save the figure to a file
plt.savefig('soc_all_storage.png')

# Optionally, display the plot if running in a notebook
plt.show()


# Prepare data for energy flow plotting
energy_flow_results = {
    'Demand': [data['Demand (MW)'][t] - Unmet_Demand[t].varValue for t in range(168)],
    'Charge LDES': [Charge_LDES[t].varValue for t in range(168)],
    'Charge SDES': [Charge_SDES[t].varValue for t in range(168)],
    'Charge Hydrogen': [Charge_Hydrogen[t].varValue for t in range(168)],
    'Curtailed Energy': [
        max(0, data['Solar Generation (MW)'][t] + data['Wind Generation (MW)'][t] -
            (Discharge_LDES[t].varValue + Discharge_SDES[t].varValue + Discharge_Hydrogen[t].varValue +
             Charge_LDES[t].varValue + Charge_SDES[t].varValue + Charge_Hydrogen[t].varValue +
             (data['Demand (MW)'][t] - Unmet_Demand[t].varValue)))
        for t in range(168)
    ]
}

df_energy_flow = pd.DataFrame(energy_flow_results)

# Plot the energy flow
fig, ax = plt.subplots(figsize=(14, 8))

# Create a stacked bar chart
df_energy_flow.plot(kind='bar', stacked=True, ax=ax, width=1)

# Customize the plot
ax.set_title('Energy Flow at Each Interval')
ax.set_xlabel('Hour')
ax.set_ylabel('Energy (MW)')
ax.legend(loc='upper right')
ax.grid(True, axis='y')

plt.tight_layout()

# Save the figure to a file
plt.savefig('energy_flow.png')

# Optionally, display the plot if running in a notebook
plt.show()