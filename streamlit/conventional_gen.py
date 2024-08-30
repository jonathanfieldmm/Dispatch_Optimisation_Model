import pandas as pd
from pulp import LpVariable, lpSum

def add_conventional_generation(prob, data, time_horizon, capacity_gas, capacity_coal, capacity_nuclear, capacity_hydro ):
   

    # Operational constraints
    min_up_time = 3  # Minimum up-time in hours
    min_down_time = 3  # Minimum down-time in hours
    ramp_rate = 10  # MW per hour

    # Costs (example values)
    cost_gas = 50  # £/MWh
    cost_coal = 60  # £/MWh
    cost_nuclear = 70  # £/MWh
    cost_hydro = 40  # £/MWh

    # Define decision variables for conventional generation
    Gen_Gas = LpVariable.dicts("Gen_Gas", range(time_horizon), lowBound=0, upBound=capacity_gas, cat='Continuous')
    Gen_Coal = LpVariable.dicts("Gen_Coal", range(time_horizon), lowBound=0, upBound=capacity_coal, cat='Continuous')
    Gen_Nuclear = LpVariable.dicts("Gen_Nuclear", range(time_horizon), lowBound=0, upBound=capacity_nuclear, cat='Continuous')
    Gen_Hydro = LpVariable.dicts("Gen_Hydro", range(time_horizon), lowBound=0, upBound=capacity_hydro, cat='Continuous')

    # Add conventional generation to the objective function
    prob += lpSum([
        cost_gas * Gen_Gas[t] +
        cost_coal * Gen_Coal[t] +
        cost_nuclear * Gen_Nuclear[t] +
        cost_hydro * Gen_Hydro[t]
        for t in range(time_horizon)
    ]), "Conventional_Generation_Cost"

    # Ramp rate constraints
    for t in range(1, time_horizon):
        prob += Gen_Gas[t] - Gen_Gas[t-1] <= ramp_rate, f"Gas_Ramp_Up_{t}"
        prob += Gen_Gas[t-1] - Gen_Gas[t] <= ramp_rate, f"Gas_Ramp_Down_{t}"
        prob += Gen_Coal[t] - Gen_Coal[t-1] <= ramp_rate, f"Coal_Ramp_Up_{t}"
        prob += Gen_Coal[t-1] - Gen_Coal[t] <= ramp_rate, f"Coal_Ramp_Down_{t}"
        prob += Gen_Hydro[t] - Gen_Hydro[t-1] <= ramp_rate, f"Hydro_Ramp_Up_{t}"
        prob += Gen_Hydro[t-1] - Gen_Hydro[t] <= ramp_rate, f"Hydro_Ramp_Down_{t}"

    # Add minimum up/down time constraints (this is a simplified version)
    for t in range(min_up_time, time_horizon):
        prob += lpSum([Gen_Gas[i] for i in range(t - min_up_time, t)]) >= capacity_gas, f"Gas_Min_Up_{t}"
        prob += lpSum([Gen_Coal[i] for i in range(t - min_up_time, t)]) >= capacity_coal, f"Coal_Min_Up_{t}"

    for t in range(min_down_time, time_horizon):
        prob += lpSum([capacity_gas - Gen_Gas[i] for i in range(t - min_down_time, t)]) >= 0, f"Gas_Min_Down_{t}"
        prob += lpSum([capacity_coal - Gen_Coal[i] for i in range(t - min_down_time, t)]) >= 0, f"Coal_Min_Down_{t}"

    return Gen_Gas, Gen_Coal, Gen_Nuclear, Gen_Hydro

def add_demand_response(prob, data, time_horizon):
    # Define demand response potential (e.g., load that can be shifted)
    max_shift = 10  # MW per hour that can be shifted

    # Define decision variables for demand response
    Shift_Down = LpVariable.dicts("Shift_Down", range(time_horizon), lowBound=0, upBound=max_shift, cat='Continuous')
    Shift_Up = LpVariable.dicts("Shift_Up", range(time_horizon), lowBound=0, upBound=max_shift, cat='Continuous')

    # Modify the demand-supply balance constraint to include demand response
    for t in range(time_horizon):
        prob += (
            Shift_Up[t] - Shift_Down[t] <= max_shift
        ), f"Demand_Response_Balance_{t}"

        prob += (
            data['Demand (MW)'][t] - Shift_Down[t] + Shift_Up[t]
        ) <= data['Demand (MW)'][t], f"Demand_Response_{t}"

    return Shift_Down, Shift_Up
