import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

def display_results(time_horizon, data, Discharge_LDES, Discharge_SDES, Discharge_Hydrogen, 
                    Gen_Gas, Gen_Coal, Gen_Nuclear, Gen_Hydro, Unmet_Demand, Curtailment, 
                    Charge_LDES, Charge_SDES, Charge_Hydrogen):
    # Display how each hour of demand is met, including all relevant components
    for t in range(time_horizon):
        st.write(f"Hour {t}:")
        st.write(f"  Solar Generation: {data['Solar Generation (MW)'][t]:.2f} MW")
        st.write(f"  Wind Generation: {data['Wind Generation (MW)'][t]:.2f} MW")
        st.write(f"  LDES Discharge: {Discharge_LDES[t].varValue:.2f} MW")
        st.write(f"  SDES Discharge: {Discharge_SDES[t].varValue:.2f} MW")
        st.write(f"  Hydrogen Discharge: {Discharge_Hydrogen[t].varValue:.2f} MW")
        st.write(f"  Gas Generation: {Gen_Gas[t].varValue:.2f} MW")
        st.write(f"  Coal Generation: {Gen_Coal[t].varValue:.2f} MW")
        st.write(f"  Nuclear Generation: {Gen_Nuclear[t].varValue:.2f} MW")
        st.write(f"  Hydro Generation: {Gen_Hydro[t].varValue:.2f} MW")
        st.write(f"  Unmet Demand: {Unmet_Demand[t].varValue:.2f} MW")
        st.write(f"  Curtailment: {Curtailment[t].varValue:.2f} MW")
        st.write(f"  LDES Charge: {Charge_LDES[t].varValue:.2f} MW")
        st.write(f"  SDES Charge: {Charge_SDES[t].varValue:.2f} MW")
        st.write(f"  Hydrogen Charge: {Charge_Hydrogen[t].varValue:.2f} MW")
        st.write(f"  Total Demand: {data['Demand (MW)'][t]:.2f} MW")
        total_generation = (data['Solar Generation (MW)'][t] + data['Wind Generation (MW)'][t] +
                            Gen_Gas[t].varValue + Gen_Coal[t].varValue + Gen_Nuclear[t].varValue + Gen_Hydro[t].varValue)
        total_discharge = (Discharge_LDES[t].varValue + Discharge_SDES[t].varValue + Discharge_Hydrogen[t].varValue)
        total_charge = (Charge_LDES[t].varValue + Charge_SDES[t].varValue + Charge_Hydrogen[t].varValue)
        total_curtailment = Curtailment[t].varValue
        total_unmet = Unmet_Demand[t].varValue
        balance_check = total_generation + total_discharge - total_charge - total_curtailment + total_unmet
        st.write(f"  Balance Check (Generation + Discharge - Charge - Curtailment + Unmet): {balance_check:.2f} MW\n")

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

    st.pyplot(fig)

def plot_soc(time_horizon, SOC_LDES, SOC_SDES, SOC_Hydrogen):
    # Prepare data for SOC plotting
    soc_results = {
        'LDES SOC': [SOC_LDES[t].varValue for t in range(1, time_horizon)],
        'SDES SOC': [SOC_SDES[t].varValue for t in range(1, time_horizon)],
        'Hydrogen SOC': [SOC_Hydrogen[t].varValue for t in range(1, time_horizon)]
    }

    df_soc_results = pd.DataFrame(soc_results)

    # Plot the SOCs on the same graph
    fig, ax = plt.subplots(figsize=(14, 8))

    ax.plot(df_soc_results.index, df_soc_results['LDES SOC'], label='LDES SOC', color='blue', linewidth=2)
    ax.plot(df_soc_results.index, df_soc_results['SDES SOC'], label='SDES SOC', color='green', linewidth=2)
    ax.plot(df_soc_results.index, df_soc_results['Hydrogen SOC'], label='Hydrogen SOC', color='red', linewidth=2)

    # Customize the plot
    ax.set_title('State of Charge (SOC) for All Storage Systems')
    ax.set_xlabel('Hour')
    ax.set_ylabel('State of Charge (MWh)')
    ax.legend(loc='upper right')
    ax.grid(True, axis='y')

    plt.tight_layout()

    st.pyplot(fig)

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

    st.pyplot(fig)



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
                                   Charge_Hydrogen[t].varValue - Curtailment[t].varValue + Unmet_Demand[t].varValue),
            'Solar Cost (£)': solar_cost,
            'Wind Cost (£)': wind_cost,
            'LDES Charge Cost (£)': ldes_charge_cost,
            'LDES Discharge Cost (£)': ldes_discharge_cost,
            'SDES Charge Cost (£)': sdes_charge_cost,
            'SDES Discharge Cost (£)': sdes_discharge_cost,
            'Hydrogen Charge Cost (£)': hydrogen_charge_cost,
            'Hydrogen Discharge Cost (£)': hydrogen_discharge_cost,
            'Unmet Demand Cost (£)': unmet_demand_cost,
            'Curtailment Cost (£)': curtailment_cost,
            'Total Cost (£)': (solar_cost + wind_cost + ldes_charge_cost + ldes_discharge_cost +
                               sdes_charge_cost + sdes_discharge_cost + hydrogen_charge_cost + 
                               hydrogen_discharge_cost + unmet_demand_cost + curtailment_cost)
        })

    average_cost = total_cost / time_horizon
    report_summary = {
        'Total Solar Generation (MWh)': [sum([entry['Solar Generation (MW)'] for entry in report])],
        'Total Wind Generation (MWh)': [sum([entry['Wind Generation (MW)'] for entry in report])],
        'Total LDES Discharge (MWh)': [sum([entry['LDES Discharge (MW)'] for entry in report])],
        'Total SDES Discharge (MWh)': [sum([entry['SDES Discharge (MW)'] for entry in report])],
        'Total Hydrogen Discharge (MWh)': [sum([entry['Hydrogen Discharge (MW)'] for entry in report])],
        'Total Gas Generation (MWh)': [sum([entry['Gas Generation (MW)'] for entry in report])],
        'Total Coal Generation (MWh)': [sum([entry['Coal Generation (MW)'] for entry in report])],
        'Total Nuclear Generation (MWh)': [sum([entry['Nuclear Generation (MW)'] for entry in report])],
        'Total Hydro Generation (MWh)': [sum([entry['Hydro Generation (MW)'] for entry in report])],
        'Total Unmet Demand (MWh)': [sum([entry['Unmet Demand (MW)'] for entry in report])],
        'Total Curtailment (MWh)': [sum([entry['Curtailment (MW)'] for entry in report])],
        'Total LDES Charge (MWh)': [sum([entry['LDES Charge (MW)'] for entry in report])],
        'Total SDES Charge (MWh)': [sum([entry['SDES Charge (MW)'] for entry in report])],
        'Total Hydrogen Charge (MWh)': [sum([entry['Hydrogen Charge (MW)'] for entry in report])],
        'Total Generation (MWh)': [sum([entry['Total Generation (MW)'] for entry in report])],
        'Total Demand (MWh)': [sum([entry['Total Demand (MW)'] for entry in report])],
        'Average Cost (£/MWh)': [average_cost],
        'Total Cost (£)': [total_cost]
    }

    df_report = pd.DataFrame(report)
    df_summary = pd.DataFrame(report_summary)
    
    with pd.ExcelWriter(filename) as writer:
        df_report.to_excel(writer, sheet_name='Detailed Report', index=False)
        df_summary.to_excel(writer, sheet_name='Summary', index=False)

    print(f"Report generated and saved to {filename}")




def display_overview(time_horizon, data, Discharge_LDES, Discharge_SDES, Discharge_Hydrogen, 
                    Gen_Gas, Gen_Coal, Gen_Nuclear, Gen_Hydro, Unmet_Demand, Curtailment, 
                    Charge_LDES, Charge_SDES, Charge_Hydrogen, cost_solar, cost_wind, 
                    cost_ldes_charge, cost_ldes_discharge, cost_sdes_charge, cost_sdes_discharge, 
                    cost_hydrogen_charge, cost_hydrogen_discharge, cost_unmet_demand, cost_curtailment):
    """
    Displays an overview of the key metrics such as total generation, unmet demand, and average cost.
    """
    # Calculate total generation for each source
    total_solar_gen = data['Solar Generation (MW)'].sum()
    total_wind_gen = data['Wind Generation (MW)'].sum()
    total_ldes_discharge = sum(Discharge_LDES[t].varValue for t in range(time_horizon))
    total_sdes_discharge = sum(Discharge_SDES[t].varValue for t in range(time_horizon))
    total_hydrogen_discharge = sum(Discharge_Hydrogen[t].varValue for t in range(time_horizon))
    total_gas_gen = sum(Gen_Gas[t].varValue for t in range(time_horizon))
    total_coal_gen = sum(Gen_Coal[t].varValue for t in range(time_horizon))
    total_nuclear_gen = sum(Gen_Nuclear[t].varValue for t in range(time_horizon))
    total_hydro_gen = sum(Gen_Hydro[t].varValue for t in range(time_horizon))
    total_unmet_demand = sum(Unmet_Demand[t].varValue for t in range(time_horizon))
    total_curtailment = sum(Curtailment[t].varValue for t in range(time_horizon))

    # Calculate total cost
    total_cost = sum(
        cost_solar * data['Solar Generation (MW)'][t] +
        cost_wind * data['Wind Generation (MW)'][t] +
        cost_ldes_charge * Charge_LDES[t].varValue +
        cost_ldes_discharge * Discharge_LDES[t].varValue +
        cost_sdes_charge * Charge_SDES[t].varValue +
        cost_sdes_discharge * Discharge_SDES[t].varValue +
        cost_hydrogen_charge * Charge_Hydrogen[t].varValue +
        cost_hydrogen_discharge * Discharge_Hydrogen[t].varValue +
        cost_unmet_demand * Unmet_Demand[t].varValue +
        cost_curtailment * Curtailment[t].varValue
        for t in range(time_horizon)
    )

    # Calculate average cost per MWh
    total_generation = (
        total_solar_gen + total_wind_gen + total_ldes_discharge + 
        total_sdes_discharge + total_hydrogen_discharge + 
        total_gas_gen + total_coal_gen + total_nuclear_gen + total_hydro_gen
    )
    average_cost_per_mwh = total_cost / total_generation if total_generation > 0 else 0

    # Display metrics
    st.subheader("Overview")
    st.write(f"Total Solar Generation: {total_solar_gen:.2f} MWh")
    st.write(f"Total Wind Generation: {total_wind_gen:.2f} MWh")
    st.write(f"Total LDES Discharge: {total_ldes_discharge:.2f} MWh")
    st.write(f"Total SDES Discharge: {total_sdes_discharge:.2f} MWh")
    st.write(f"Total Hydrogen Discharge: {total_hydrogen_discharge:.2f} MWh")
    st.write(f"Total Gas Generation: {total_gas_gen:.2f} MWh")
    st.write(f"Total Coal Generation: {total_coal_gen:.2f} MWh")
    st.write(f"Total Nuclear Generation: {total_nuclear_gen:.2f} MWh")
    st.write(f"Total Hydro Generation: {total_hydro_gen:.2f} MWh")
    st.write(f"Total Unmet Demand: {total_unmet_demand:.2f} MWh")
    st.write(f"Total Curtailment: {total_curtailment:.2f} MWh")
    st.write(f"Total Cost: £{total_cost:.2f}")
    st.write(f"Average Cost per MWh: £{average_cost_per_mwh:.2f}")

def create_hourly_breakdown(time_horizon, data, Discharge_LDES, Discharge_SDES, Discharge_Hydrogen, 
                            Gen_Gas, Gen_Coal, Gen_Nuclear, Gen_Hydro, Unmet_Demand, Curtailment, 
                            Charge_LDES, Charge_SDES, Charge_Hydrogen):
    """
    Creates a DataFrame with the hourly breakdown of the energy system.
    """
    hourly_data = []
    for t in range(time_horizon):
        row = {
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
            'Total Demand (MW)': data['Demand (MW)'][t],
            'Total Generation (MW)': (
                data['Solar Generation (MW)'][t] + data['Wind Generation (MW)'][t] +
                Gen_Gas[t].varValue + Gen_Coal[t].varValue + Gen_Nuclear[t].varValue +
                Gen_Hydro[t].varValue + Discharge_LDES[t].varValue + 
                Discharge_SDES[t].varValue + Discharge_Hydrogen[t].varValue
            ),
            'Balance Check (MW)': (
                data['Solar Generation (MW)'][t] + data['Wind Generation (MW)'][t] +
                Discharge_LDES[t].varValue + Discharge_SDES[t].varValue + 
                Discharge_Hydrogen[t].varValue + Gen_Gas[t].varValue + 
                Gen_Coal[t].varValue + Gen_Nuclear[t].varValue + Gen_Hydro[t].varValue - 
                Charge_LDES[t].varValue - Charge_SDES[t].varValue - 
                Charge_Hydrogen[t].varValue - Curtailment[t].varValue + 
                Unmet_Demand[t].varValue
            )
        }
        hourly_data.append(row)
    
    df_hourly = pd.DataFrame(hourly_data)
    return df_hourly
