import streamlit as st
import streamlit as st
import pandas as pd
import results_and_plotting2 as rp
import optimisation as opt
import time

st.title("Dispatch Optimisation Modelling")

# Sidebar for input parameters
st.sidebar.header("Input Parameters")

# Group inputs into sections for better organization
st.sidebar.subheader("Upload Data Files")
# File uploader for demand profile
demand_file = st.sidebar.file_uploader("Upload Demand Profile", type=["xlsx"])
if demand_file is not None:
    demand_profile = pd.read_excel(demand_file)
else:
    st.error("Please upload a demand profile file.")

# File uploader for generation profiles
generation_file = st.sidebar.file_uploader("Upload Generation Profile", type=["xlsx"])
if generation_file is not None:
    generation_profiles = pd.read_excel(generation_file)
else:
    st.error("Please upload a generation profile file.")

if demand_file is not None and generation_file is not None:
    st.sidebar.subheader("Simulation Settings")
    # Number of weeks
    number_of_weeks = st.sidebar.slider("Number of Weeks", min_value=1, max_value=52, value=1)

    # Extend the data for the number of weeks
    demand_profile = pd.concat([demand_profile] * number_of_weeks, ignore_index=True)
    generation_profiles = pd.concat([generation_profiles] * number_of_weeks, ignore_index=True)

    st.sidebar.subheader("Generation Capacities (MW)")
    # Generation capacities
    capacities = {
        'wind': st.sidebar.number_input("Capacity of Wind (MW)", min_value=0.0, value=20.0),
        'solar': st.sidebar.number_input("Capacity of Solar (MW)", min_value=0.0, value=20.0),
        'gas': st.sidebar.number_input("Capacity of Gas (MW)", min_value=0.0, value=20.0),
        'coal': st.sidebar.number_input("Capacity of Coal (MW)", min_value=0.0, value=20.0),
        'nuclear': st.sidebar.number_input("Capacity of Nuclear (MW)", min_value=0.0, value=20.0),
        'hydro': st.sidebar.number_input("Capacity of Hydro (MW)", min_value=0.0, value=20.0),
        'ldes': st.sidebar.number_input("Capacity of LDES (MWh)", min_value=0.0, value=1000.0),
        'sdes': st.sidebar.number_input("Capacity of SDES (MWh)", min_value=0.0, value=500.0),
        'hydrogen': st.sidebar.number_input("Capacity of Hydrogen Storage (MWh)", min_value=0.0, value=1500.0)
    }

    st.sidebar.subheader("Storage Efficiencies")
    # Storage efficiencies
    efficiencies = {
        'ldes': st.sidebar.number_input("Efficiency of LDES", min_value=0.0, max_value=1.0, value=0.85),
        'sdes': st.sidebar.number_input("Efficiency of SDES", min_value=0.0, max_value=1.0, value=0.9),
        'hydrogen': st.sidebar.number_input("Efficiency of Hydrogen Storage", min_value=0.0, max_value=1.0, value=0.75)
    }

    st.sidebar.subheader("Cost Inputs (£/MWh)")
    # Cost inputs
    costs = {
        'solar': st.sidebar.number_input("Cost of Solar", min_value=0.0, value=10.0),
        'wind': st.sidebar.number_input("Cost of Wind", min_value=0.0, value=15.0),
        'gas': st.sidebar.number_input("Cost of Gas", min_value=0.0, value=50.0),
        'coal': st.sidebar.number_input("Cost of Coal", min_value=0.0, value=60.0),
        'nuclear': st.sidebar.number_input("Cost of Nuclear", min_value=0.0, value=70.0),
        'hydro': st.sidebar.number_input("Cost of Hydro", min_value=0.0, value=40.0),
        'ldes_charge': st.sidebar.number_input("Cost of LDES Charge", min_value=0.0, value=5.0),
        'ldes_discharge': st.sidebar.number_input("Cost of LDES Discharge", min_value=0.0, value=7.0),
        'sdes_charge': st.sidebar.number_input("Cost of SDES Charge", min_value=0.0, value=3.0),
        'sdes_discharge': st.sidebar.number_input("Cost of SDES Discharge", min_value=0.0, value=5.0),
        'hydrogen_charge': st.sidebar.number_input("Cost of Hydrogen Charge", min_value=0.0, value=8.0),
        'hydrogen_discharge': st.sidebar.number_input("Cost of Hydrogen Discharge", min_value=0.0, value=10.0),
        'unmet_demand': st.sidebar.number_input("Cost of Unmet Demand", min_value=0.0, value=100.0),
        'curtailment': st.sidebar.number_input("Cost of Curtailment", min_value=0.0, value=5.0)
    }

    if st.button('Run Simulation'):
        with st.spinner('Running the optimization...'):
            time.sleep(1)  # Simulate delay for progress indication

            # Calculate the actual generation profiles by multiplying capacity factors by the defined capacities
            solar_generation = generation_profiles['Solar Capacity Factor'] * capacities['solar']
            wind_generation = generation_profiles['Wind Capacity Factor'] * capacities['wind']

            # Extract the demand profile
            demand = demand_profile['Demand']

            # Ensure the data is in the correct format (168 * number_of_weeks hourly intervals)
            time_horizon = 168 * number_of_weeks
            data = pd.DataFrame({
                'Solar Generation (MW)': solar_generation,
                'Wind Generation (MW)': wind_generation,
                'Demand (MW)': demand
            })

            # Run the optimization
            prob, status, decision_vars = opt.run_optimization(
                data, time_horizon, costs, capacities, efficiencies
            )

            st.success(f"Optimization completed with status: {status}")

            # Unpack the decision variables dictionary for easier access in results and plotting
            Charge_LDES = decision_vars['Charge_LDES']
            Discharge_LDES = decision_vars['Discharge_LDES']
            Charge_SDES = decision_vars['Charge_SDES']
            Discharge_SDES = decision_vars['Discharge_SDES']
            Charge_Hydrogen = decision_vars['Charge_Hydrogen']
            Discharge_Hydrogen = decision_vars['Discharge_Hydrogen']
            Unmet_Demand = decision_vars['Unmet_Demand']
            Curtailment = decision_vars['Curtailment']
            SOC_LDES = decision_vars['SOC_LDES']
            SOC_SDES = decision_vars['SOC_SDES']
            SOC_Hydrogen = decision_vars['SOC_Hydrogen']
            Gen_Gas = decision_vars['Gen_Gas']
            Gen_Coal = decision_vars['Gen_Coal']
            Gen_Nuclear = decision_vars['Gen_Nuclear']
            Gen_Hydro = decision_vars['Gen_Hydro']

            # Display results in tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Hourly Breakdown", "Demand vs Supply", "SOC", "Energy Flow"])

            with tab1:
                st.write("Overview")
                rp.display_overview(time_horizon, data, Discharge_LDES, Discharge_SDES, Discharge_Hydrogen, 
                                    Gen_Gas, Gen_Coal, Gen_Nuclear, Gen_Hydro, Unmet_Demand, Curtailment, 
                                    Charge_LDES, Charge_SDES, Charge_Hydrogen, costs['solar'], costs['wind'], 
                                    costs['ldes_charge'], costs['ldes_discharge'], costs['sdes_charge'], 
                                    costs['sdes_discharge'], costs['hydrogen_charge'], costs['hydrogen_discharge'], 
                                    costs['unmet_demand'], costs['curtailment'])

            with tab2:
                st.write("Hourly Breakdown")
                hourly_df = rp.create_hourly_breakdown(time_horizon, data, Discharge_LDES, Discharge_SDES, Discharge_Hydrogen, 
                                                       Gen_Gas, Gen_Coal, Gen_Nuclear, Gen_Hydro, Unmet_Demand, Curtailment, 
                                                       Charge_LDES, Charge_SDES, Charge_Hydrogen)
                st.dataframe(hourly_df)

            with tab3:
                st.write("Demand vs Supply")
                rp.plot_results(time_horizon, data, Discharge_LDES, Discharge_SDES, Discharge_Hydrogen, 
                                Gen_Gas, Gen_Coal, Gen_Nuclear, Gen_Hydro, Unmet_Demand)

            with tab4:
                st.write("State of Charge (SOC) for Storage Systems")
                rp.plot_soc(time_horizon, SOC_LDES, SOC_SDES, SOC_Hydrogen)

            with tab5:
                st.write("Energy Flow")
                rp.plot_energy_flow(time_horizon, data, Discharge_LDES, Discharge_SDES, Discharge_Hydrogen, 
                                    Gen_Gas, Gen_Coal, Gen_Nuclear, Gen_Hydro, Charge_LDES, Charge_SDES, 
                                    Charge_Hydrogen, Unmet_Demand)

            # Button to download the report
            if st.button('Download Report'):
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
                    cost_solar=costs['solar'],
                    cost_wind=costs['wind'],
                    cost_ldes_charge=costs['ldes_charge'],
                    cost_ldes_discharge=costs['ldes_discharge'],
                    cost_sdes_charge=costs['sdes_charge'],
                    cost_sdes_discharge=costs['sdes_discharge'],
                    cost_hydrogen_charge=costs['hydrogen_charge'],
                    cost_hydrogen_discharge=costs['hydrogen_discharge'],
                    cost_unmet_demand=costs['unmet_demand'],
                    cost_curtailment=costs['curtailment']
                )
                st.success('Report generated: optimization_report.xlsx')
