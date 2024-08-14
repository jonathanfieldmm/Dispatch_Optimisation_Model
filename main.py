import pandas as pd
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus
import numpy as np
import matplotlib.pyplot as plt

import conventional_gen as cg
import results_and_plotting as rp
# Define the number of weeks
number_of_weeks = 1

# Load the demand profile and generation profiles from the Excel files
demand_profile = pd.read_excel('demand_profile.xlsx', header=0)
generation_profiles = pd.read_excel('generation_profiles.xlsx', header=0)

# Extend the data for the number of weeks
demand_profile = pd.concat([demand_profile] * number_of_weeks, ignore_index=True)
generation_profiles = pd.concat([generation_profiles] * number_of_weeks, ignore_index=True)

# Define capacities for wind and solar (in MW)
capacity_wind = 20
capacity_solar = 20

# Calculate the actual generation profiles by multiplying capacity factors by the defined capacities
solar_generation = generation_profiles['Solar Capacity Factor'] * capacity_solar
wind_generation = generation_profiles['Wind Capacity Factor'] * capacity_wind

# Extract the demand profile
demand = demand_profile['Demand']

# Ensure the data is in the correct format (168 * number_of_weeks hourly intervals)
time_horizon = 168 * number_of_weeks
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

# Add conventional generation to the optimization problem
Gen_Gas, Gen_Coal, Gen_Nuclear, Gen_Hydro = cg.add_conventional_generation(prob, data, time_horizon)

# Add demand response to the optimization problem
Shift_Down, Shift_Up = cg.add_demand_response(prob, data, time_horizon)
# Solve the optimization problem

# Define decision variables
Charge_LDES = LpVariable.dicts("Charge_LDES", range(time_horizon), lowBound=0, cat='Continuous')
Discharge_LDES = LpVariable.dicts("Discharge_LDES", range(time_horizon), lowBound=0, cat='Continuous')

Charge_SDES = LpVariable.dicts("Charge_SDES", range(time_horizon), lowBound=0, cat='Continuous')
Discharge_SDES = LpVariable.dicts("Discharge_SDES", range(time_horizon), lowBound=0, cat='Continuous')

Charge_Hydrogen = LpVariable.dicts("Charge_Hydrogen", range(time_horizon), lowBound=0, cat='Continuous')
Discharge_Hydrogen = LpVariable.dicts("Discharge_Hydrogen", range(time_horizon), lowBound=0, cat='Continuous')

Unmet_Demand = LpVariable.dicts("Unmet_Demand", range(time_horizon), lowBound=0, cat='Continuous')
Curtailment = LpVariable.dicts("Curtailment", range(time_horizon), lowBound=0, cat='Continuous')

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
    cost_curtailment * Curtailment[t]
    for t in range(time_horizon)
]), "Total_Cost"

# Set the initial state of charge (SOC) for each storage type to zero
SOC_LDES = {0: 0}
SOC_SDES = {0: 0}
SOC_Hydrogen = {0: 0}

# Additional constraint to ensure storage discharge only happens to meet demand
for t in range(1, time_horizon):
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
for t in range(time_horizon):
    prob += (
        data['Solar Generation (MW)'][t] +
        data['Wind Generation (MW)'][t] +
        Discharge_LDES[t] +
        Discharge_SDES[t] +
        Discharge_Hydrogen[t] +
        Gen_Gas[t] +
        Gen_Coal[t] +
        Gen_Nuclear[t] +
        Gen_Hydro[t] +
        Unmet_Demand[t]
        == data['Demand (MW)'][t] +
        Charge_LDES[t] +
        Charge_SDES[t] +
        Charge_Hydrogen[t] +
        Curtailment[t]  # Add curtailment to ensure excess generation is handled
    ), f"Demand_Supply_Constraint_{t}"

# Final state of charge must be within capacity limits
prob += SOC_LDES[time_horizon-1] <= capacity_ldes, "Final_LDES_SOC"
prob += SOC_SDES[time_horizon-1] <= capacity_sdes, "Final_SDES_SOC"
prob += SOC_Hydrogen[time_horizon-1] <= capacity_hydrogen, "Final_Hydrogen_SOC"

# Solve the optimization problem
prob.solve()

print("Status:", LpStatus[prob.status])

# Call functions from results_and_plotting.py
rp.display_results(time_horizon, data, Discharge_LDES, Discharge_SDES, Discharge_Hydrogen, 
                   Gen_Gas, Gen_Coal, Gen_Nuclear, Gen_Hydro, Unmet_Demand, Curtailment, 
                   Charge_LDES, Charge_SDES, Charge_Hydrogen)

rp.plot_results(time_horizon, data, Discharge_LDES, Discharge_SDES, Discharge_Hydrogen, 
                Gen_Gas, Gen_Coal, Gen_Nuclear, Gen_Hydro, Unmet_Demand)

rp.plot_soc(time_horizon, SOC_LDES, SOC_SDES, SOC_Hydrogen)

rp.plot_energy_flow(time_horizon, data, Discharge_LDES, Discharge_SDES, Discharge_Hydrogen, 
                    Gen_Gas, Gen_Coal, Gen_Nuclear, Gen_Hydro, Charge_LDES, Charge_SDES, 
                    Charge_Hydrogen, Unmet_Demand)
# Generate the comprehensive report
rp.generate_report(
    filename='optimization_report.xlsx',
    time_horizon=time_horizon,
    data=data,
    Discharge_LDES=Discharge_LDES,
    Discharge_SDES=Discharge_SDES,
    Discharge_Hydrogen=Discharge_Hydrogen,
    Gen_Gas=Gen_Gas,
    Gen_Coal=Gen_Coal,
    Gen_Nuclear=Gen_Nuclear,
    Gen_Hydro=Gen_Hydro,
    Unmet_Demand=Unmet_Demand,
    Curtailment=Curtailment,
    Charge_LDES=Charge_LDES,
    Charge_SDES=Charge_SDES,
    Charge_Hydrogen=Charge_Hydrogen,
    cost_solar=cost_solar,
    cost_wind=cost_wind,
    cost_ldes_charge=cost_ldes_charge,
    cost_ldes_discharge=cost_ldes_discharge,
    cost_sdes_charge=cost_sdes_charge,
    cost_sdes_discharge=cost_sdes_discharge,
    cost_hydrogen_charge=cost_hydrogen_charge,
    cost_hydrogen_discharge=cost_hydrogen_discharge,
    cost_unmet_demand=cost_unmet_demand,
    cost_curtailment=cost_curtailment
)