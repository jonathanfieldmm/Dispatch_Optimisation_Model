import pandas as pd
import matplotlib.pyplot as plt

def display_results(time_horizon, data, Discharge_LDES, Discharge_SDES, Discharge_Hydrogen, 
                    Gen_Gas, Gen_Coal, Gen_Nuclear, Gen_Hydro, Unmet_Demand, Curtailment, 
                    Charge_LDES, Charge_SDES, Charge_Hydrogen):
    # Display how each hour of demand is met, including all relevant components
    for t in range(time_horizon):
        print(f"Hour {t}:")
        print(f"  Solar Generation: {data['Solar Generation (MW)'][t]:.2f} MW")
        print(f"  Wind Generation: {data['Wind Generation (MW)'][t]:.2f} MW")
        print(f"  LDES Discharge: {Discharge_LDES[t].varValue:.2f} MW")
        print(f"  SDES Discharge: {Discharge_SDES[t].varValue:.2f} MW")
        print(f"  Hydrogen Discharge: {Discharge_Hydrogen[t].varValue:.2f} MW")
        print(f"  Gas Generation: {Gen_Gas[t].varValue:.2f} MW")
        print(f"  Coal Generation: {Gen_Coal[t].varValue:.2f} MW")
        print(f"  Nuclear Generation: {Gen_Nuclear[t].varValue:.2f} MW")
        print(f"  Hydro Generation: {Gen_Hydro[t].varValue:.2f} MW")
        print(f"  Unmet Demand: {Unmet_Demand[t].varValue:.2f} MW")
        print(f"  Curtailment: {Curtailment[t].varValue:.2f} MW")
        print(f"  LDES Charge: {Charge_LDES[t].varValue:.2f} MW")
        print(f"  SDES Charge: {Charge_SDES[t].varValue:.2f} MW")
        print(f"  Hydrogen Charge: {Charge_Hydrogen[t].varValue:.2f} MW")
        print(f"  Total Demand: {data['Demand (MW)'][t]:.2f} MW")
        total_generation = (data['Solar Generation (MW)'][t] + data['Wind Generation (MW)'][t] +
                            Gen_Gas[t].varValue + Gen_Coal[t].varValue + Gen_Nuclear[t].varValue + Gen_Hydro[t].varValue)
        total_discharge = (Discharge_LDES[t].varValue + Discharge_SDES[t].varValue + Discharge_Hydrogen[t].varValue)
        total_charge = (Charge_LDES[t].varValue + Charge_SDES[t].varValue + Charge_Hydrogen[t].varValue)
        total_curtailment = Curtailment[t].varValue
        total_unmet = Unmet_Demand[t].varValue
        balance_check = total_generation + total_discharge - total_charge - total_curtailment + total_unmet
        print(f"  Balance Check (Generation + Discharge - Charge - Curtailment + Unmet): {balance_check:.2f} MW\n")

def plot_results(time_horizon, data, Discharge_LDES, Discharge_SDES, Discharge_Hydrogen, 
                 Gen_Gas, Gen_Coal, Gen_Nuclear, Gen_Hydro, Unmet_Demand):
    # Prepare data for plotting
    results = {
        'Solar Generation': [data['Solar Generation (MW)'][t] for t in range(time_horizon)],
        'Wind Generation': [data['Wind Generation (MW)'][t] for t in range(time_horizon)],
        'LDES Discharge': [Discharge_LDES[t].varValue for t in range(time_horizon)],
        'SDES Discharge': [Discharge_SDES[t].varValue for t in range(time_horizon)],
        'Hydrogen Discharge': [Discharge_Hydrogen[t].varValue for t in range(time_horizon)],
        'Gas Generation': [Gen_Gas[t].varValue for t in range(time_horizon)],
        'Coal Generation': [Gen_Coal[t].varValue for t in range(time_horizon)],
        'Nuclear Generation': [Gen_Nuclear[t].varValue for t in range(time_horizon)],
        'Hydro Generation': [Gen_Hydro[t].varValue for t in range(time_horizon)],
        'Unmet Demand': [Unmet_Demand[t].varValue for t in range(time_horizon)]
    }

    df_results = pd.DataFrame(results)

    # Plot the stacked bar chart
    fig, ax = plt.subplots(figsize=(14, 8))

    # Create a stacked bar chart
    df_results.plot(kind='bar', stacked=True, ax=ax, width=1)

    # Plot the demand line
    ax.plot(df_results.index, data['Demand (MW)'][:time_horizon], label='Demand', linestyle='--', color='black', linewidth=2)

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

def plot_soc(time_horizon, SOC_LDES, SOC_SDES, SOC_Hydrogen):
    # Prepare data for SOC plotting
    soc_results = {
        'LDES SOC': [SOC_LDES[t].varValue for t in range(1, time_horizon)],
        'SDES SOC': [SOC_SDES[t].varValue for t in range(1, time_horizon)],
        'Hydrogen SOC': [SOC_Hydrogen[t].varValue for t in range(1, time_horizon)]
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

def plot_energy_flow(time_horizon, data, Discharge_LDES, Discharge_SDES, Discharge_Hydrogen, 
                     Gen_Gas, Gen_Coal, Gen_Nuclear, Gen_Hydro, Charge_LDES, Charge_SDES, Charge_Hydrogen, Unmet_Demand):
    # Prepare data for energy flow plotting
    energy_flow_results = {
        'Demand': [data['Demand (MW)'][t] - Unmet_Demand[t].varValue for t in range(time_horizon)],
        'Charge LDES': [Charge_LDES[t].varValue for t in range(time_horizon)],
        'Charge SDES': [Charge_SDES[t].varValue for t in range(time_horizon)],
        'Charge Hydrogen': [Charge_Hydrogen[t].varValue for t in range(time_horizon)],
        'Curtailed Energy': [
            max(0, data['Solar Generation (MW)'][t] + data['Wind Generation (MW)'][t] +
                Gen_Gas[t].varValue + Gen_Coal[t].varValue + Gen_Nuclear[t].varValue + Gen_Hydro[t].varValue -
                (Discharge_LDES[t].varValue + Discharge_SDES[t].varValue + Discharge_Hydrogen[t].varValue +
                 Charge_LDES[t].varValue + Charge_SDES[t].varValue + Charge_Hydrogen[t].varValue +
                 (data['Demand (MW)'][t] - Unmet_Demand[t].varValue)))
            for t in range(time_horizon)
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


def generate_report(filename, time_horizon, data, Discharge_LDES, Discharge_SDES, Discharge_Hydrogen, 
                    Gen_Gas, Gen_Coal, Gen_Nuclear, Gen_Hydro, Unmet_Demand, Curtailment, 
                    Charge_LDES, Charge_SDES, Charge_Hydrogen, cost_solar, cost_wind, 
                    cost_ldes_charge, cost_ldes_discharge, cost_sdes_charge, cost_sdes_discharge, 
                    cost_hydrogen_charge, cost_hydrogen_discharge, cost_unmet_demand, cost_curtailment):
    report = []

    total_cost = 0
    for t in range(time_horizon):
        solar_cost = cost_solar * data['Solar Generation (MW)'][t]
        wind_cost = cost_wind * data['Wind Generation (MW)'][t]
        ldes_charge_cost = cost_ldes_charge * Charge_LDES[t].varValue
        ldes_discharge_cost = cost_ldes_discharge * Discharge_LDES[t].varValue
        sdes_charge_cost = cost_sdes_charge * Charge_SDES[t].varValue
        sdes_discharge_cost = cost_sdes_discharge * Discharge_SDES[t].varValue
        hydrogen_charge_cost = cost_hydrogen_charge * Charge_Hydrogen[t].varValue
        hydrogen_discharge_cost = cost_hydrogen_discharge * Discharge_Hydrogen[t].varValue
        unmet_demand_cost = cost_unmet_demand * Unmet_Demand[t].varValue
        curtailment_cost = cost_curtailment * Curtailment[t].varValue

        total_cost += (solar_cost + wind_cost + ldes_charge_cost + ldes_discharge_cost +
                       sdes_charge_cost + sdes_discharge_cost + hydrogen_charge_cost + 
                       hydrogen_discharge_cost + unmet_demand_cost + curtailment_cost)

        report.append({
            'Hour': t,
            'Solar Generation (MW)': data['Solar Generation (MW)'][t],
            'Wind Generation (MW)': data['Wind Generation (MW)'][t],
            'LDES Discharge (MW)': Discharge_LDES[t].varValue,
            'SDES Discharge (MW)': Discharge_SDES[t].varValue,
            'Hydrogen Discharge (MW)': Discharge_Hydrogen[t].varValue,
            'Gas Generation (MW)': Gen_Gas[t].varValue,
            'Coal Generation (MW)': Gen_Coal[t].varValue,
            'Nuclear Generation (MW)': Gen_Nuclear[t].varValue,
            'Hydro Generation (MW)': Gen_Hydro[t].varValue,
            'Unmet Demand (MW)': Unmet_Demand[t].varValue,
            'Curtailment (MW)': Curtailment[t].varValue,
            'LDES Charge (MW)': Charge_LDES[t].varValue,
            'SDES Charge (MW)': Charge_SDES[t].varValue,
            'Hydrogen Charge (MW)': Charge_Hydrogen[t].varValue,
            'Total Generation (MW)': (data['Solar Generation (MW)'][t] + data['Wind Generation (MW)'][t] +
                                       Gen_Gas[t].varValue + Gen_Coal[t].varValue + Gen_Nuclear[t].varValue + 
                                       Gen_Hydro[t].varValue),
            'Total Demand (MW)': data['Demand (MW)'][t],
            'Balance Check (MW)': (data['Solar Generation (MW)'][t] + data['Wind Generation (MW)'][t] +
                                   Discharge_LDES[t].varValue + Discharge_SDES[t].varValue + Discharge_Hydrogen[t].varValue +
                                   Gen_Gas[t].varValue + Gen_Coal[t].varValue + Gen_Nuclear[t].varValue + 
                                   Gen_Hydro[t].varValue - Charge_LDES[t].varValue - Charge_SDES[t].varValue - 
                                   Charge_Hydrogen[t].varValue - Curtailment[t].varValue + Unmet_Demand[t].varValue)
        })

    average_cost = total_cost / time_horizon
    report.append({'Average Cost (£/MWh)': average_cost})
    
    df_report = pd.DataFrame(report)
    
    with pd.ExcelWriter(filename) as writer:
        df_report.to_excel(writer, sheet_name='Detailed Report', index=False)
        
        summary = {
            'Total Solar Generation (MWh)': [df_report['Solar Generation (MW)'].sum()],
            'Total Wind Generation (MWh)': [df_report['Wind Generation (MW)'].sum()],
            'Total LDES Discharge (MWh)': [df_report['LDES Discharge (MW)'].sum()],
            'Total SDES Discharge (MWh)': [df_report['SDES Discharge (MW)'].sum()],
            'Total Hydrogen Discharge (MWh)': [df_report['Hydrogen Discharge (MW)'].sum()],
            'Total Gas Generation (MWh)': [df_report['Gas Generation (MW)'].sum()],
            'Total Coal Generation (MWh)': [df_report['Coal Generation (MW)'].sum()],
            'Total Nuclear Generation (MWh)': [df_report['Nuclear Generation (MW)'].sum()],
            'Total Hydro Generation (MWh)': [df_report['Hydro Generation (MW)'].sum()],
            'Total Unmet Demand (MWh)': [df_report['Unmet Demand (MW)'].sum()],
            'Total Curtailment (MWh)': [df_report['Curtailment (MW)'].sum()],
            'Total LDES Charge (MWh)': [df_report['LDES Charge (MW)'].sum()],
            'Total SDES Charge (MWh)': [df_report['SDES Charge (MW)'].sum()],
            'Total Hydrogen Charge (MWh)': [df_report['Hydrogen Charge (MW)'].sum()],
            'Total Generation (MWh)': [df_report['Total Generation (MW)'].sum()],
            'Total Demand (MWh)': [df_report['Total Demand (MW)'].sum()],
            'Average Cost (£/MWh)': [average_cost]
        }
        
        df_summary = pd.DataFrame(summary)
        df_summary.to_excel(writer, sheet_name='Summary', index=False)

    print(f"Report generated and saved to {filename}")