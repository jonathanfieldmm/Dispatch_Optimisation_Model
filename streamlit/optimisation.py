import pandas as pd
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus

def run_optimization(data, time_horizon, costs, capacities, efficiencies):
    
    # Create the LP problem
    prob = LpProblem("Least_Cost_Dispatch", LpMinimize)

    # Add conventional generation to the optimisation problem
    Gen_Gas, Gen_Coal, Gen_Nuclear, Gen_Hydro = add_conventional_generation(
        prob, data, time_horizon, capacities, costs
    )

    # Add demand response to the optimisation problem
    Shift_Down, Shift_Up = add_demand_response(prob, data, time_horizon)

    # Define decision variables
    decision_vars = {
        'Charge_LDES': LpVariable.dicts("Charge_LDES", range(time_horizon), lowBound=0, cat='Continuous'),
        'Discharge_LDES': LpVariable.dicts("Discharge_LDES", range(time_horizon), lowBound=0, cat='Continuous'),
        'Charge_SDES': LpVariable.dicts("Charge_SDES", range(time_horizon), lowBound=0, cat='Continuous'),
        'Discharge_SDES': LpVariable.dicts("Discharge_SDES", range(time_horizon), lowBound=0, cat='Continuous'),
        'Charge_Hydrogen': LpVariable.dicts("Charge_Hydrogen", range(time_horizon), lowBound=0, cat='Continuous'),
        'Discharge_Hydrogen': LpVariable.dicts("Discharge_Hydrogen", range(time_horizon), lowBound=0, cat='Continuous'),
        'Unmet_Demand': LpVariable.dicts("Unmet_Demand", range(time_horizon), lowBound=0, cat='Continuous'),
        'Curtailment': LpVariable.dicts("Curtailment", range(time_horizon), lowBound=0, cat='Continuous'),
        'SOC_LDES': {0: 0},
        'SOC_SDES': {0: 0},
        'SOC_Hydrogen': {0: 0},
        'Gen_Gas': Gen_Gas,
        'Gen_Coal': Gen_Coal,
        'Gen_Nuclear': Gen_Nuclear,
        'Gen_Hydro': Gen_Hydro,
        'Shift_Down': Shift_Down,
        'Shift_Up': Shift_Up
    }
    # Planning a network model: Cost per MWh of generation etc
    # Objective: Minimise the total cost of generation, charging, discharging, and unmet demand
    prob += lpSum([
        costs['solar'] * data['Solar Generation (MW)'][t] +
        costs['wind'] * data['Wind Generation (MW)'][t] +
        costs['ldes_charge'] * decision_vars['Charge_LDES'][t] +
        costs['ldes_discharge'] * decision_vars['Discharge_LDES'][t] +
        costs['sdes_charge'] * decision_vars['Charge_SDES'][t] +
        costs['sdes_discharge'] * decision_vars['Discharge_SDES'][t] +
        costs['hydrogen_charge'] * decision_vars['Charge_Hydrogen'][t] +
        costs['hydrogen_discharge'] * decision_vars['Discharge_Hydrogen'][t] +
        costs['gas'] * decision_vars['Gen_Gas'][t] +
        costs['coal'] * decision_vars['Gen_Coal'][t] +
        costs['nuclear'] * decision_vars['Gen_Nuclear'][t] +
        costs['hydro'] * decision_vars['Gen_Hydro'][t] +
        costs['unmet_demand'] * decision_vars['Unmet_Demand'][t] +
        costs['curtailment'] * decision_vars['Curtailment'][t]
        for t in range(time_horizon)
    ]), "Total_Cost"

    # Additional constraint to ensure storage discharge only happens to meet demand
    for t in range(1, time_horizon):
        decision_vars['SOC_LDES'][t] = LpVariable(f"SOC_LDES_{t}", lowBound=0, upBound=capacities['ldes'])
        decision_vars['SOC_SDES'][t] = LpVariable(f"SOC_SDES_{t}", lowBound=0, upBound=capacities['sdes'])
        decision_vars['SOC_Hydrogen'][t] = LpVariable(f"SOC_Hydrogen_{t}", lowBound=0, upBound=capacities['hydrogen'])

        # Storage dynamics
        prob += decision_vars['SOC_LDES'][t] == decision_vars['SOC_LDES'][t-1] + decision_vars['Charge_LDES'][t] * efficiencies['ldes'] - decision_vars['Discharge_LDES'][t] * (1 / efficiencies['ldes']), f"LDES_SOC_{t}"
        prob += decision_vars['SOC_SDES'][t] == decision_vars['SOC_SDES'][t-1] + decision_vars['Charge_SDES'][t] * efficiencies['sdes'] - decision_vars['Discharge_SDES'][t] * (1 / efficiencies['sdes']), f"SDES_SOC_{t}"
        prob += decision_vars['SOC_Hydrogen'][t] == decision_vars['SOC_Hydrogen'][t-1] + decision_vars['Charge_Hydrogen'][t] * efficiencies['hydrogen'] - decision_vars['Discharge_Hydrogen'][t] * (1 / efficiencies['hydrogen']), f"Hydrogen_SOC_{t}"

        # Ensure that the discharge from storage does not exceed the available SOC at that time
        prob += decision_vars['Discharge_LDES'][t] <= decision_vars['SOC_LDES'][t-1], f"LDES_Discharge_Limit_{t}"
        prob += decision_vars['Discharge_SDES'][t] <= decision_vars['SOC_SDES'][t-1], f"SDES_Discharge_Limit_{t}"
        prob += decision_vars['Discharge_Hydrogen'][t] <= decision_vars['SOC_Hydrogen'][t-1], f"Hydrogen_Discharge_Limit_{t}"

        # Discharge can only happen to meet demand
        prob += decision_vars['Discharge_LDES'][t] <= data['Demand (MW)'][t], f"LDES_Discharge_Meets_Demand_{t}"
        prob += decision_vars['Discharge_SDES'][t] <= data['Demand (MW)'][t], f"SDES_Discharge_Meets_Demand_{t}"
        prob += decision_vars['Discharge_Hydrogen'][t] <= data['Demand (MW)'][t], f"Hydrogen_Discharge_Meets_Demand_{t}"

        # Capacity constraints
        prob += decision_vars['SOC_LDES'][t] <= capacities['ldes'], f"LDES_Capacity_{t}"
        prob += decision_vars['SOC_SDES'][t] <= capacities['sdes'], f"SDES_Capacity_{t}"
        prob += decision_vars['SOC_Hydrogen'][t] <= capacities['hydrogen'], f"Hydrogen_Capacity_{t}"

    # Demand-Supply Constraint with Unmet Demand and Curtailed Energy
    for t in range(time_horizon):
        prob += (
            data['Solar Generation (MW)'][t] +
            data['Wind Generation (MW)'][t] +
            decision_vars['Discharge_LDES'][t] +
            decision_vars['Discharge_SDES'][t] +
            decision_vars['Discharge_Hydrogen'][t] +
            decision_vars['Gen_Gas'][t] +
            decision_vars['Gen_Coal'][t] +
            decision_vars['Gen_Nuclear'][t] +
            decision_vars['Gen_Hydro'][t] +
            decision_vars['Unmet_Demand'][t]
            == data['Demand (MW)'][t] +
            decision_vars['Charge_LDES'][t] +
            decision_vars['Charge_SDES'][t] +
            decision_vars['Charge_Hydrogen'][t] +
            decision_vars['Curtailment'][t]  # Add curtailment to ensure excess generation is handled
        ), f"Demand_Supply_Constraint_{t}"

    # Final state of charge must be within capacity limits
    prob += decision_vars['SOC_LDES'][time_horizon-1] <= capacities['ldes'], "Final_LDES_SOC"
    prob += decision_vars['SOC_SDES'][time_horizon-1] <= capacities['sdes'], "Final_SDES_SOC"
    prob += decision_vars['SOC_Hydrogen'][time_horizon-1] <= capacities['hydrogen'], "Final_Hydrogen_SOC"

    # Solve the optimization problem
    prob.solve()

    return prob, LpStatus[prob.status], decision_vars

def add_conventional_generation(prob, data, time_horizon, capacities, costs):
    # Define decision variables for conventional generation
    Gen_Gas = LpVariable.dicts("Gen_Gas", range(time_horizon), lowBound=0, upBound=capacities['gas'], cat='Continuous')
    Gen_Coal = LpVariable.dicts("Gen_Coal", range(time_horizon), lowBound=0, upBound=capacities['coal'], cat='Continuous')
    Gen_Nuclear = LpVariable.dicts("Gen_Nuclear", range(time_horizon), lowBound=0, upBound=capacities['nuclear'], cat='Continuous')
    Gen_Hydro = LpVariable.dicts("Gen_Hydro", range(time_horizon), lowBound=0, upBound=capacities['hydro'], cat='Continuous')

    # Add conventional generation to the objective function
    prob += lpSum([
        costs['gas'] * Gen_Gas[t] +
        costs['coal'] * Gen_Coal[t] +
        costs['nuclear'] * Gen_Nuclear[t] +
        costs['hydro'] * Gen_Hydro[t]
        for t in range(time_horizon)
    ]), "Conventional_Generation_Cost"

    # Operational constraints
    min_up_time = 3  # Minimum up-time in hours
    min_down_time = 3  # Minimum down-time in hours
    ramp_rate = 10  # MW per hour

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
        prob += lpSum([Gen_Gas[i] for i in range(t - min_up_time, t)]) >= capacities['gas'], f"Gas_Min_Up_{t}"
        prob += lpSum([Gen_Coal[i] for i in range(t - min_up_time, t)]) >= capacities['coal'], f"Coal_Min_Up_{t}"

    for t in range(min_down_time, time_horizon):
        prob += lpSum([capacities['gas'] - Gen_Gas[i] for i in range(t - min_down_time, t)]) >= 0, f"Gas_Min_Down_{t}"
        prob += lpSum([capacities['coal'] - Gen_Coal[i] for i in range(t - min_down_time, t)]) >= 0, f"Coal_Min_Down_{t}"

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



