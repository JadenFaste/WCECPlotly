# Things to plot for heating:
# EP_Total_HVAC_Power_W, T_Outdoor_ecobee_F, T_HeatSetPoint_F, T_Thermostat_F, and TP_Capacity_Heating_Btuh (if there is no entry for a certain index, TE_Capacity_Heating_Btu is on at the same time,
# so write a check if TP_Capacity_Heating_Btuh is 0, then check if TE_Capacity_Heating_Btu is 0 at the time too, and then mark that empty entry as 0

import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import spearmanr
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from datetime import datetime
import os

cwd = os.getcwd()
print(cwd)
path = r'C:\Users\bober\OneDrive\Documents\GitHub\WCECPlotly'
os.chdir(path)

# Load the CSV data
# file_path = "https://raw.githubusercontent.com/JadenFaste/WCECPlotly/main/Test%20data.csv"
file_path = "test_data_updated.csv"
filepath_2 = "https://raw.githubusercontent.com/JadenFaste/WCECPlotly/main/Villara%203%20Function%20Cycle%20Data.csv"
df = pd.read_csv(file_path)
df_2 = pd.read_csv(filepath_2)
data = pd.read_csv('cycle_data_updated.csv')
data['idx_Cycle_Start'] = pd.to_datetime(data['idx_Cycle_Start'], errors='coerce')
data['Year'] = data['idx_Cycle_Start'].dt.year

def create_static_histogram(filtered_data, x_column, title, label):
    fig = px.histogram(
        filtered_data, 
        x=x_column, 
        title=title,
        labels={x_column: label, 'count': 'Frequency'}, 
        color='Year',
        nbins=50,
        opacity=0.6
    )
    
    # Get the color sequence from Plotly Express
    color_sequence = px.colors.qualitative.Plotly
    
    # Calculate and add yearly means and mean lines for 2022, 2023, and 2024
    for i, year in enumerate([2022, 2023, 2024]):
        year_data = filtered_data[filtered_data['Year'] == year]
        mean_value = year_data[x_column].mean()
        if not pd.isna(mean_value):
            # Add annotation for the mean value
            fig.add_annotation(
                x=0.01,
                y=0.95 - 0.05 * i,  # Stagger annotations
                xref='paper',
                yref='paper',
                text=f"Mean ({year}): {mean_value:.2f} W",
                showarrow=False,
                font=dict(color=color_sequence[i], size=12),
                align='left'
            )
            # Add a vertical dotted line at the mean value
            fig.add_shape(
                type="line",
                x0=mean_value,
                y0=0,
                x1=mean_value,
                y1=1,
                xref='x',
                yref='paper',
                line=dict(
                    color=color_sequence[i],
                    width=2,
                    dash="dot",
                ),
            )
    
    # Update layout to set the title font size
    fig.update_layout(
        title_font=dict(size=10)  # Adjust the font size as needed
    )
    
    return fig

# Create static histograms for each condition
ac_mode_data = data[(data['AC_Mode'] == 1) & (data['AC_and_DHW_Mode'] == 0)]
fig_ac_mode = create_static_histogram(ac_mode_data, 'EP_OutdoorUnit_PowerSum_W', 'Frequency of Outdoor Unit Power Sum by Year (AC Mode On, AC and DHW Mode Off)', 'Outdoor Unit Power Sum (W)')

space_heat_data = data[(data['Space_Heat_Mode'] == 1) & (data['Defrost_Mode'] == 0)]
fig_space_heat = create_static_histogram(space_heat_data, 'EP_OutdoorUnit_PowerSum_W', 'Frequency of Outdoor Unit Power Sum by Year (Space Heat Mode On, Defrost Mode Off)', 'Outdoor Unit Power Sum (W)')

water_heating_data = data[
    (data['Water_Heating_Mode'] == 1) &
    (data['AC_and_DHW_Mode'] == 0) &
    (data['Defrost_Mode'] == 0) &
    (data['Space_Heat_Mode'] == 0) &
    ~((data['Year'].isin([2022, 2023])) & (data['EP_OutdoorUnit_PowerSum_W'] <= 400))
]
fig_water_heating = create_static_histogram(water_heating_data, 'EP_OutdoorUnit_PowerSum_W', 'Frequency of Outdoor Unit Power Sum by Year (Water Heating Mode On, No AC/DHW, Defrost, or Space Heat Mode)', 'Outdoor Unit Power Sum (W)')

ah_power_ac_data = data[(data['AC_Mode'] == 1) & (data['AC_and_DHW_Mode'] == 0)]
fig_ah_power_ac = create_static_histogram(ah_power_ac_data, 'EP_AH_PowerSum_W', 'Frequency of Air Handler Power Sum by Year (AC Mode On, AC and DHW Mode Off)', 'Air Handler Power Sum (W)')

ah_power_space_heat_data = data[(data['Space_Heat_Mode'] == 1) & (data['Defrost_Mode'] == 0)]
fig_ah_power_space_heat = create_static_histogram(ah_power_space_heat_data, 'EP_AH_PowerSum_W', 'Frequency of Air Handler Power Sum by Year (Space Heat Mode On, Defrost Mode Off)', 'Air Handler Power Sum (W)')

ah_power_water_heating_data = data[
    (data['Water_Heating_Mode'] == 1) &
    (data['AC_and_DHW_Mode'] == 0) &
    (data['Defrost_Mode'] == 0) &
    (data['Space_Heat_Mode'] == 0)
]
fig_ah_power_water_heating = create_static_histogram(ah_power_water_heating_data, 'EP_AH_PowerSum_W', 'Frequency of Air Handler Power Sum by Year (Water Heating Mode On, No AC/DHW, Defrost, or Space Heat Mode)', 'Air Handler Power Sum (W)')

# Convert the 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y %H:%M')

# Avg of t2 and t3
df['T_HotTank_T2_T3_avg_F'] = (df['T_HotTank_T2_F'] + df['T_HotTank_T3_F']) / 2

# Convert 'idx_Mid_Cycle' to datetime
data['idx_Mid_Cycle'] = pd.to_datetime(data['idx_Mid_Cycle'], errors='coerce')
data['Year'] = data['idx_Mid_Cycle'].dt.year

# Filter cycle data
filtered_df = data[(data['AC_Mode'] == 1) & (data['AC_and_DHW_Mode'] == 0) & (data['COP_Steady_State'] > 0.2)]

# Convert 'Year' to string to ensure it is treated as a categorical variable
filtered_df['Year'] = filtered_df['Year'].astype(str)

# Filter necessary columns and drop NaN
filtered_df = filtered_df[['T_Outdoor_ecobee_F', 'COP_Steady_State', 'idx_Mid_Cycle', 'Year']].dropna()

# Variables available for plotting (excluding the Date column for the dropdown)
available_variables = df.columns.drop('Date')

binary_columns = ['AC_Mode', 'AC_and_DHW_Mode', 'Space_Heat_Mode', 'Water_Heating_Mode', 'Defrost_Mode']

mode_colors = {
    'Controller_AC_Mode': 'CadetBlue',
    'Controller_AC_and_DHW_Mode': 'LightSkyBlue',
    'Controller_Defrost_Mode': 'LightCoral',
    'Controller_Space_Heat_Mode': 'LightGreen',
    'Controller_Water_Heating_Mode': 'LightSalmon'
}

def get_all_shaded_regions(df, columns):
    """Get shaded regions for all columns."""
    all_shapes = {}
    for col in columns:
        shapes = []
        in_shaded_region = False
        start_date = None
        for i, row in df.iterrows():
            if row[col] == 1 and not in_shaded_region:
                in_shaded_region = True
                start_date = row['Date']
            elif row[col] == 0 and in_shaded_region:
                in_shaded_region = False
                end_date = row['Date']
                shapes.append(
                    dict(
                        type="rect",
                        xref="x",
                        yref="paper",
                        x0=start_date,
                        y0=0,
                        x1=end_date,
                        y1=1,
                        fillcolor=mode_colors[col],
                        opacity=0.5,
                        layer="below",
                        line_width=0,
                    )
                )
        if in_shaded_region:
            shapes.append(
                dict(
                    type="rect",
                    xref="x",
                    yref="paper",
                    x0=start_date,
                    y0=0,
                    x1=df['Date'].iloc[-1],
                    y1=1,
                    fillcolor=mode_colors[col],
                    opacity=0.5,
                    layer="below",
                    line_width=0,
                )
            )
        all_shapes[col] = shapes
    return all_shapes

# Create Gantt chart
tasks = pd.DataFrame({
    'Task': [
        'Baseline Tango 1', 'Static Summer Price Signal Tango', 'Baseline Tango 2 (Away from home) ?', 'Static Summer Price & GHG Signal Tango (Away from home) ?', 'Baseline Tango 3', '2 Week moving price signal Tango', 'Baseline Tango 4',
        'Baseline Apt C 1', 'Static Summer Price Signal Apt C', 'Baseline Apt C 2', 'Static Summer Price & GHG Signal Apt C', 'Baseline C Apt 3', '2 Week moving price signal Apt C', 'Baseline Apt C 4', 
        'Baseline Apt D 1', 'Static Summer Price Signal Apt D', 'Baseline Apt D 1', 'Static Summer Price & GHG Signal Apt D', 'Baseline Apt D 3', '2 Week moving price signal Apt D',
        'Test 1', 'Test 2', 'Test 3', 'Test 4'
    ],
    'Start': [
        pd.Timestamp('2024-07-24'), pd.Timestamp('2024-07-31'), pd.Timestamp('2024-08-07'), pd.Timestamp('2024-08-14'), pd.Timestamp('2024-08-29'), pd.Timestamp('2024-09-05'), pd.Timestamp('2024-09-19'),
        pd.Timestamp('2024-07-24'), pd.Timestamp('2024-07-31'), pd.Timestamp('2024-08-07'), pd.Timestamp('2024-08-13'), pd.Timestamp('2024-08-26'), pd.Timestamp('2024-09-02'), pd.Timestamp('2024-09-16'),
        pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.NaT,
        pd.Timestamp('2024-06-12'), pd.Timestamp('2024-06-12'), pd.Timestamp('2024-06-12'), pd.Timestamp('2024-06-12')
    ],
    'End': [
        pd.Timestamp('2024-07-30'), pd.Timestamp('2024-08-06'), pd.Timestamp('2024-08-13'), pd.Timestamp('2024-08-20'), pd.Timestamp('2024-09-04'), pd.Timestamp('2024-09-18'), pd.Timestamp('2024-09-25'),
        pd.Timestamp('2024-07-30'), pd.Timestamp('2024-08-06'), pd.Timestamp('2024-08-13'), pd.Timestamp('2024-08-20'), pd.Timestamp('2024-09-01'), pd.Timestamp('2024-09-15'), pd.Timestamp('2024-09-22'),
        pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.NaT,
        pd.Timestamp('2024-06-15'), pd.Timestamp('2024-06-15'), pd.Timestamp('2024-06-15'), pd.Timestamp('2024-06-15')
    ],
    'Group': [
        'WCEC Tango Site', 'WCEC Tango Site', 'WCEC Tango Site', 'WCEC Tango Site', 'WCEC Tango Site', 'WCEC Tango Site', 'WCEC Tango Site',
        'WCEC Bear Creek Apt C', 'WCEC Bear Creek Apt C', 'WCEC Bear Creek Apt C', 'WCEC Bear Creek Apt C', 'WCEC Bear Creek Apt C', 'WCEC Bear Creek Apt C', 'WCEC Bear Creek Apt C',
        'WCEC Bear Creek Apt D', 'WCEC Bear Creek Apt D', 'WCEC Bear Creek Apt D', 'WCEC Bear Creek Apt D', 'WCEC Bear Creek Apt D', 'WCEC Bear Creek Apt D',
        'Other Tests', 'Other Tests', 'Other Tests', 'Other Tests'
    ]
})

fig_gantt = px.timeline(tasks, x_start="Start", x_end="End", y="Task", color="Group")

fig_gantt.update_yaxes(categoryorder="array", categoryarray=tasks['Task'].tolist())

current_date = datetime.now().strftime('%Y-%m-%d')
fig_gantt.add_vline(x=current_date, line_width=3, line_dash="dash", line_color="red")

fig_gantt.update_layout(
    title="Project Timeline", 
    xaxis_title="Date", 
    yaxis_title="Task",
    width=2400,
    height=800
)

# Create the Dash app
#######################################################################################################
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div([
    html.H1("Time Series Dashboard"),

    html.Div([
        html.Label("Select Date Range:"),
        dcc.DatePickerRange(
            id='date-picker-range',
            start_date='2024-08-17',  
            end_date='2024-08-18',
            display_format='MM/DD/YYYY'
        )
    ], style={'margin-top': '20px', 'margin-bottom': '20px'}),

    # Dropdown for selecting the primary variable
    html.Div([
        html.Label("Select Primary Variable:"),
        dcc.Dropdown(
            id='primary-yaxis-column-name',
            options=[{'label': i, 'value': i} for i in available_variables],
            value=available_variables[0]  # Default value
        ),
    ]),

    # Dropdown for selecting the secondary variable
    html.Div([
        html.Label("Select Secondary Variable:"),
        dcc.Dropdown(
            id='secondary-yaxis-column-name',
            options=[{'label': i, 'value': i} for i in available_variables],
            value=available_variables[1]  # Default value
        ),
    ]),

    # Checkboxes for binary columns
    html.Div([
        html.Label("Select Modes to Highlight:"),
        dcc.Checklist(
            id='mode-selector',
            options=[
                {'label': 'AC Mode', 'value': 'Controller_AC_Mode'},
                {'label': 'AC and DHW Mode', 'value': 'Controller_AC_and_DHW_Mode'},
                {'label': 'Defrost Mode', 'value': 'Controller_Defrost_Mode'},
                {'label': 'Space Heat Mode', 'value': 'Controller_Space_Heat_Mode'},
                {'label': 'Water Heating Mode', 'value': 'Controller_Water_Heating_Mode'}
            ],
            value=['Controller_Water_Heating_Mode'],  # Default value
            inline=True
        ),
    ], style={'margin-top': '20px', 'margin-bottom': '20px'}),

    html.Hr(),

    # line chart
    html.H4("Choose two variables to plot"),
    dcc.Graph(id='line-plot'),
    html.Div(id='line-plot-missing-data'),

    # Separator
    html.Hr(),

    # Second Graph
    html.H4("Fixed plot for water heater operations"),
    dcc.Graph(id='fixed-variables-plot'),
    html.Div(id='fixed-variables-missing-data'),

    # Separator
    html.Hr(),

    # Third Graph
    html.H4("Fixed plot for HVAC cooling operations"),
    dcc.Graph(id='custom-variables-plot'),
    html.Div(id='custom-variables-missing-data'),

    html.Hr(),
    html.H4("Fixed plot for HVAC heating operations"),
    dcc.Graph(id='duplicate-custom-variables-plot'),

    html.Hr(),  # Separator
    html.H4("Histogram for key variables"),
    html.Div([
        html.Div([dcc.Graph(id='hot-tank-t2-t3-histogram')],
                 style={'width': '500px', 'display': 'inline-block', 'margin': '0 5px'}),
        html.Div([dcc.Graph(id='outdoor-ecobee-histogram')],
                 style={'width': '500px', 'display': 'inline-block', 'margin': '0 5px'}),
        html.Div([dcc.Graph(id='indoor-temperature-histogram')],
                 style={'width': '500px', 'display': 'inline-block', 'margin': '0 5px'}),
        html.Div([dcc.Graph(id='water-draw-histogram')],
                 style={'width': '500px', 'display': 'inline-block', 'margin': '0 5px'})
    ], style={'display': 'flex', 'justify-content': 'space-around'}),
    html.Div(id='energy-sum-display-fig2', style={'textAlign': 'center', 'margin-top': '20px'}),
    html.Div(id='water-draw-sum-display-fig2', style={'textAlign': 'center', 'margin-top': '20px'}),

    # Separator
    html.Hr(),

    dcc.Dropdown(
        id='binary-column-selector',
        options=[{'label': col, 'value': col} for col in binary_columns],
        value=binary_columns[0]
    ),

    # Regression
    dcc.Graph(id='graph'),
    html.Label([
        "Set number of features",
        dcc.Slider(id='PolyFeat',
                   min=1,
                   max=3,
                   step=1,
                   marks={i: '{}'.format(i) for i in range(10)},
                   value=1,
                   )
    ]),
    html.Div(id='equation'),

    html.Hr(),
    html.H4("Gantt Chart for project timeline"),
    dcc.Graph(id='gantt-chart', figure=fig_gantt),
    
    html.Hr(),
    html.H4("Scatterplot for COP vs. Outdoor Temperature grouped by year, with current date highlighted"),
    dcc.Graph(id='scatter-plot'),

    html.Div([
        dcc.Graph(id='ac-mode-plot', figure=fig_ac_mode, style={'display': 'inline-block', 'width': '32%'}),
        dcc.Graph(id='space-heat-plot', figure=fig_space_heat, style={'display': 'inline-block', 'width': '32%'}),
        dcc.Graph(id='water-heating-plot', figure=fig_water_heating, style={'display': 'inline-block', 'width': '32%'}),
    ], style={'display': 'flex', 'justify-content': 'space-between'}),
    html.Div([
        dcc.Graph(id='ah-power-ac-plot', figure=fig_ah_power_ac, style={'display': 'inline-block', 'width': '32%'}),
        dcc.Graph(id='ah-power-space-heat-plot', figure=fig_ah_power_space_heat, style={'display': 'inline-block', 'width': '32%'}),
        dcc.Graph(id='ah-power-water-heating-plot', figure=fig_ah_power_water_heating, style={'display': 'inline-block', 'width': '32%'}),
    ], style={'display': 'flex', 'justify-content': 'space-between'})
])

###############################################################################################
# dash app endings
@app.callback(
    [Output('graph', 'figure'),
     Output('equation', 'children')],
    [Input('PolyFeat', 'value'),
     Input('binary-column-selector', 'value')]
)
def update_figure(nFeatures, selected_column):
    print(f"Callback triggered with nFeatures = {nFeatures}, selected_column = {selected_column}")
    global model

    # data
    df_2 = pd.read_csv(filepath_2)
    filtered_df = df_2[df_2[selected_column] == 1]
    filtered_df = df_2[df_2[selected_column] == 1].copy()

    # Convert NaNs to zeros
    filtered_df.loc[:, "T_Outdoor_ecobee_F"] = pd.to_numeric(filtered_df["T_Outdoor_ecobee_F"], errors='coerce').fillna(0)
    filtered_df.loc[:, "COP"] = pd.to_numeric(filtered_df["COP"], errors='coerce').fillna(0)

    filtered_df = filtered_df[(filtered_df["COP"] <= 10) & (filtered_df["COP"] != 0) & (filtered_df["COP"] > -20) & (filtered_df["T_Outdoor_ecobee_F"] != 0)]

    y_filtered = filtered_df["COP"]
    x_filtered = filtered_df["T_Outdoor_ecobee_F"]

    # Model fitting & finding equation
    model = make_pipeline(PolynomialFeatures(nFeatures), LinearRegression())
    model.fit(np.array(x_filtered).reshape(-1, 1), y_filtered)
    linear_model = model.named_steps['linearregression']
    coefficients = linear_model.coef_
    intercept = linear_model.intercept_

    # Predict values using the fitted model
    y_predicted = model.predict(np.array(x_filtered).reshape(-1, 1))

    # Calculate Spearman's correlation for the predicted values
    spearman_corr, _ = spearmanr(y_predicted, y_filtered)

    # Generate prediction points for a smoother curve
    x_reg = np.linspace(min(x_filtered), max(x_filtered), 100)
    y_reg = model.predict(x_reg.reshape(-1, 1))
    reg_df = pd.DataFrame({'x': x_reg, 'model': y_reg})

    # Prepare data for AIC and BIC calculation
    X_design = np.array(x_filtered).reshape(-1, 1)
    X_design = PolynomialFeatures(nFeatures).fit_transform(X_design)
    model_sm = sm.OLS(y_filtered, X_design).fit()

    spearman_corr, _ = spearmanr(y_predicted, y_filtered)

    # Calculate AIC and BIC
    aic = model_sm.aic
    bic = model_sm.bic

    # Format the regression equation
    equation = f"y = {intercept:.10f}"
    for i in range(1, len(coefficients)):
        coef = coefficients[i]
        equation += f" + {coef:.10f}x^{i}"
    # Prepare the regression equation and metrics string
    equation_and_metrics = (
        f"Regression Equation: {equation}\n"
        f"Spearman's Correlation: {spearman_corr:.4f}\n"
        f"AIC: {aic:.2f}, BIC: {bic:.2f}"
    )

    # Create the figure
    fig = go.Figure()
    fig.add_traces(go.Scatter(x=x_filtered, y=y_filtered, mode='markers', name='observations'))
    reg_df = reg_df.sort_values(by=['x'])
    fig.add_traces(go.Scatter(x=reg_df['x'], y=reg_df['model'], mode='lines', name='model'))

    fig.update_layout(
        title="Polynomial Regression Analysis",
        xaxis_title="Outdoor Temperature (F)",
        yaxis_title="Coefficient of Performance (COP)"
    )

    # Return the figure and the regression equation
    return fig, equation_and_metrics

@app.callback(
    [Output('line-plot', 'figure'),
     Output('fixed-variables-plot', 'figure'),
     Output('custom-variables-plot', 'figure'),
     Output('energy-sum-display-fig2', 'children'),
     Output('line-plot-missing-data', 'children'),
     Output('fixed-variables-missing-data', 'children'),
     Output('custom-variables-missing-data', 'children'),
     Output('duplicate-custom-variables-plot', 'figure')], 
    [Input('primary-yaxis-column-name', 'value'),
     Input('secondary-yaxis-column-name', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date'),
     Input('mode-selector', 'value')]
)
def update_graph(primary_var, secondary_var, start_date, end_date, modes):
    # Filter the dataframe based on the selected date range
    filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy()

    # Ensure the relevant columns are numeric
    filtered_df['TP_Capacity_Cooling_Btuh'] = pd.to_numeric(filtered_df['TP_Capacity_Cooling_Btuh'], errors='coerce')
    filtered_df['TE_Capacity_Cooling_Btu'] = pd.to_numeric(filtered_df['TE_Capacity_Cooling_Btu'], errors='coerce')

    # Handle missing TP_Capacity_Cooling_Btuh values
    # Create a mask for rows where TP_Capacity_Cooling_Btuh is NaN
    mask = filtered_df['TP_Capacity_Cooling_Btuh'].isna()

    # Apply the conditional logic using np.where
    filtered_df.loc[mask, 'TP_Capacity_Cooling_Btuh'] = np.where(
        filtered_df.loc[mask, 'TE_Capacity_Cooling_Btu'] == 0,
        0,
        filtered_df.loc[mask, 'TE_Capacity_Cooling_Btu'] * 12
    )

    all_shapes = get_all_shaded_regions(filtered_df, modes)

    fig1 = px.line(filtered_df, x='Date', y=primary_var)
    fig1.add_traces(go.Scatter(x=filtered_df['Date'], y=filtered_df[secondary_var], mode='lines', name=secondary_var))
    for mode, shapes in all_shapes.items():
        for shape in shapes:
            fig1.add_shape(shape)

    fig1.update_layout(
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.5,
            xanchor='center',
            x=0.5
        )
    )

    # Figure 2: Fixed Variables Plot (Example)
    fixed_variables = ["EP_Total_HVAC_Power_W", "T_Outdoor_ecobee_F", "VFR_HotTank_WaterDraw_FlowRate_gpm",
                       "T_HotTank_T2_T3_avg_F"]
    temperature_variables = ["T_Outdoor_ecobee_F", "T_HotTank_T2_T3_avg_F"]
    colors = ['blue', 'red', 'green', 'purple']

    fig2 = go.Figure()
    total_energy_kwh = filtered_df['EE_Total_HVAC_Energy_kWh'].sum()
    energy_sum_text = f'Total HVAC Energy Consumption: {total_energy_kwh:.2f} kWh'

    for mode, shapes in all_shapes.items():
        for shape in shapes:
            fig2.add_shape(shape)

    non_temperature_variables = []
    for i, var in enumerate(fixed_variables):
        if var in temperature_variables:
            fig2.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df[var], mode='lines', name=var,
                                      line=dict(color=colors[i])))
        else:
            non_temperature_variables.append(var)
            yaxis = f'y{len(non_temperature_variables) + 1}'
            fig2.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df[var], mode='lines', name=var,
                                      line=dict(color=colors[i]), yaxis=yaxis))

    fig2.update_layout(
        yaxis=dict(title="Temperature (F)"),
    )
# Number of non-temperature variables
    n_non_temp_vars = len(non_temperature_variables)

# Adjust the x-axis domain to leave less space on the right for extra y-axes
    fig2.update_layout(
        xaxis=dict(domain=[0.0, 0.92])  # Increased domain to 90% of width
)

# Adjust the right margin to prevent clipping of y-axis labels
    fig2.update_layout(
        margin=dict(r=100)  # Adjusted right margin as needed
    )

# Calculate starting position and available space
    start_pos = fig2.layout.xaxis.domain[1] + 0.01  # Slightly right of the plot area
    end_pos = 1.0 - 0.000005  # Leave a small margin on the far right
    available_space = end_pos - start_pos

# Decrease delta_pos to reduce gaps between y-axes
    delta_pos = .5  # Smaller gap between y-axes

# Adjust delta_pos if necessary to ensure all y-axes fit within the figure
    max_required_space = delta_pos * (n_non_temp_vars - 1)
    if max_required_space > available_space:
        delta_pos = available_space / (n_non_temp_vars - 1)

# Update layout for each additional y-axis
    for i, var in enumerate(non_temperature_variables):
        position = start_pos + i * delta_pos
        fig2.update_layout(**{
            f'yaxis{i + 2}': dict(
                title=var,
                anchor='free',
                overlaying='y',
                side='right',
                position=position
            )
        })

    fig2.update_layout(
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.5,
            xanchor='center',
            x=0.5
        ),
    )

    # Figure 3: Custom Variables Plot (Including TP_Capacity_Cooling_Btuh)
    custom_variables = ["EP_Total_HVAC_Power_W", "T_Outdoor_ecobee_F", "T_CoolSetpoint_F", "T_Thermostat_F", "TP_Capacity_Cooling_Btuh"]
    temperature_variables_fig3 = ["T_Outdoor_ecobee_F", "T_CoolSetpoint_F", "T_Thermostat_F"]
    colors_fig3 = ['blue', 'red', 'green', 'orange', 'purple']

    fig3 = go.Figure()
    for mode, shapes in all_shapes.items():
        for shape in shapes:
            fig3.add_shape(shape)

    non_temperature_variables_fig3 = []
    for i, var in enumerate(custom_variables):
        if var in temperature_variables_fig3:
            fig3.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df[var], mode='lines', name=var,
                                      line=dict(color=colors_fig3[i])))
        else:
            non_temperature_variables_fig3.append(var)
            yaxis = f'y{len(non_temperature_variables_fig3) + 1}'
            fig3.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df[var], mode='lines', name=var,
                                      line=dict(color=colors_fig3[i]), yaxis=yaxis))

    fig3.update_layout(
        yaxis=dict(title="Temperature (F)"),
    )
# Number of non-temperature variables
    n_non_temp_vars = len(non_temperature_variables_fig3)

# Adjust the x-axis domain to leave less space on the right for extra y-axes
    fig3.update_layout(
        xaxis=dict(domain=[0.0, 0.92])  # Increased domain to 90% of width
)

# Adjust the right margin to prevent clipping of y-axis labels
    fig3.update_layout(
        margin=dict(r=100)  # Adjusted right margin as needed
    )

# Calculate starting position and available space
    start_pos = fig3.layout.xaxis.domain[1] + 0.01  # Slightly right of the plot area
    end_pos = 1.0 - 0.000005  # Leave a small margin on the far right
    available_space = end_pos - start_pos

# Decrease delta_pos to reduce gaps between y-axes
    delta_pos = .5  # Smaller gap between y-axes

# Adjust delta_pos if necessary to ensure all y-axes fit within the figure
    max_required_space = delta_pos * (n_non_temp_vars - 1)
    if max_required_space > available_space:
        delta_pos = available_space / (n_non_temp_vars - 1)

# Update layout for each additional y-axis
    for i, var in enumerate(non_temperature_variables_fig3):
        position = start_pos + i * delta_pos
        fig3.update_layout(**{
            f'yaxis{i + 2}': dict(
                title=var,
                anchor='free',
                overlaying='y',
                side='right',
                position=position
            )
        }) 

    fig3.update_layout(
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.5,
            xanchor='center',
            x=0.5
        )
    )

    # Figure 4: Additional Custom Variables Plot (Example)
    custom_variables_fig4 = ["EP_Total_HVAC_Power_W", "T_Outdoor_ecobee_F", "T_HeatSetpoint_F", "T_Thermostat_F", "TP_Capacity_Heating_Btuh"]
    temperature_variables_fig4 = ["T_Outdoor_ecobee_F", "T_HeatSetpoint_F", "T_Thermostat_F"]
    colors_fig4 = ['blue', 'red', 'green', 'orange', 'purple']

    # Handle missing TP_Capacity_Heating_Btuh values by filling NaNs with 0
    filtered_df['TP_Capacity_Heating_Btuh'] = filtered_df['TP_Capacity_Heating_Btuh'].fillna(0)

    fig4 = go.Figure()
    for mode, shapes in all_shapes.items():
        for shape in shapes:
            fig4.add_shape(shape)

    non_temperature_variables_fig4 = []
    for i, var in enumerate(custom_variables_fig4):
        if var in temperature_variables_fig4:
            fig4.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df[var], mode='lines', name=var,
                                      line=dict(color=colors_fig4[i])))
        else:
            non_temperature_variables_fig4.append(var)
            yaxis = f'y{len(non_temperature_variables_fig4) + 1}'
            fig4.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df[var], mode='lines', name=var,
                                      line=dict(color=colors_fig4[i]), yaxis=yaxis))

    fig4.update_layout(
        yaxis=dict(title="Temperature (F)"),
    )
# Number of non-temperature variables
    n_non_temp_vars = len(non_temperature_variables_fig4)

# Adjust the x-axis domain to leave less space on the right for extra y-axes
    fig4.update_layout(
        xaxis=dict(domain=[0.0, 0.92])  # Increased domain to 90% of width
)

# Adjust the right margin to prevent clipping of y-axis labels
    fig4.update_layout(
        margin=dict(r=100)  # Adjusted right margin as needed
    )

# Calculate starting position and available space
    start_pos = fig4.layout.xaxis.domain[1] + 0.01  # Slightly right of the plot area
    end_pos = 1.0 - 0.000005  # Leave a small margin on the far right
    available_space = end_pos - start_pos

# Decrease delta_pos to reduce gaps between y-axes
    delta_pos = .5  # Smaller gap between y-axes

# Adjust delta_pos if necessary to ensure all y-axes fit within the figure
    max_required_space = delta_pos * (n_non_temp_vars - 1)
    if max_required_space > available_space:
        delta_pos = available_space / (n_non_temp_vars - 1)

# Update layout for each additional y-axis
    for i, var in enumerate(non_temperature_variables_fig4):
        position = start_pos + i * delta_pos
        fig4.update_layout(**{
            f'yaxis{i + 2}': dict(
                title=var,
                anchor='free',
                overlaying='y',
                side='right',
                position=position
            )
        })

    fig4.update_layout(
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.5,
            xanchor='center',
            x=0.5
        )
    )

    # Prepare the missing data text
    def count_missing_values(df, column):
        return df[column].isna().sum()

    primary_missing_data = f"Missing data in {primary_var}: {count_missing_values(filtered_df, primary_var)}"
    secondary_missing_data = f"Missing data in {secondary_var}: {count_missing_values(filtered_df, secondary_var)}"
    fixed_missing_data = ", ".join([f"{var}: {count_missing_values(filtered_df, var)}" for var in fixed_variables])
    custom_missing_data = ", ".join([f"{var}: {count_missing_values(filtered_df, var)}" for var in custom_variables])

    return fig1, fig2, fig3, energy_sum_text, primary_missing_data + "; " + secondary_missing_data, fixed_missing_data, custom_missing_data, fig4


@app.callback(
    Output('hot-tank-t2-t3-histogram', 'figure'),
    [Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_histogram(start_date, end_date):
    filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    fig = px.histogram(filtered_df, x='T_HotTank_T2_T3_avg_F',
                       title='T_HotTank_T2_T3_avg_F',
                       labels={'T_HotTank_T2_T3_avg_F': 'Temperature'})
    fig.update_layout(height=300, width=400)
    return fig

@app.callback(
    Output('outdoor-ecobee-histogram', 'figure'),
    [Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_outdoor_histogram(start_date, end_date):
    filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    fig = px.histogram(filtered_df, x='T_Outdoor_ecobee_F',
                       title='T_Outdoor_ecobee_F',
                       labels={'T_Outdoor_ecobee_F': 'Outdoor Temperature'})
    fig.update_layout(width=400, height=300)
    return fig

@app.callback(
    Output('indoor-temperature-histogram', 'figure'),
    [Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_indoor_histogram(start_date, end_date):
    filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    fig = px.histogram(filtered_df, x='T_Thermostat_F',
                       title='T_Thermostat_F',
                       labels={'T_Thermostat_F': 'Indoor Temperature'})
    fig.update_layout(width=400, height=300)
    return fig

@app.callback(
    Output('water-draw-histogram', 'figure'),
    [Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_water_draw_histogram(start_date, end_date):
    filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    fig = px.histogram(filtered_df, x='VFR_HotTank_WaterDraw_FlowRate_gpm',
                       title='VFR_HotTank_WaterDraw_FlowRate_gpm',
                       labels={'VFR_HotTank_WaterDraw_FlowRate_gpm': 'Flow Rate'})
    fig.update_layout(width=400, height=300)
    return fig

@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_graph(start_date, end_date):
    # Convert start and end dates to datetime
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Create a new column to track whether the data point is within the selected date range
    filtered_df['color'] = filtered_df.apply(
        lambda row: 'red' if start_date <= row['idx_Mid_Cycle'] <= end_date else row['Year'], axis=1
    )
    
    # Create the scatter plot
    fig = px.scatter(
        filtered_df,
        x='T_Outdoor_ecobee_F',
        y='COP_Steady_State',
        color='color',
        color_discrete_map={
            '2022': 'greenyellow', '2023': 'lightblue', '2024': 'khaki', 'red': 'red'
        },
        labels={'T_Outdoor_ecobee_F': 'T_Outdoor_ecobee_F', 'COP_Steady_State': 'COP Steady State'},
        title='COP Steady State vs Outdoor Temperature by Year'
    )
    
    # Add grid and customize layout
    fig.update_layout(
        xaxis_title='T_Outdoor_ecobee_F',
        yaxis_title='COP Steady State',
        legend_title='Year',
        template='plotly_white'
    )
    
    # Update marker size
    fig.update_traces(marker=dict(size=8))  # Adjust the size value as needed

    return fig

@app.callback(
    [Output('ac-mode-plot', 'figure'),
     Output('space-heat-plot', 'figure'),
     Output('water-heating-plot', 'figure'),
     Output('ah-power-ac-plot', 'figure'),
     Output('ah-power-space-heat-plot', 'figure'),
     Output('ah-power-water-heating-plot', 'figure')],
    [Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_figures(start_date, end_date):
    # Filter data based on the selected date range
    mask = (data['idx_Cycle_Start'] >= start_date) & (data['idx_Cycle_Start'] <= end_date)
    date_filtered_data = data[mask]
    
    # Calculate means for each condition
    means = {}
    means['Outdoor Unit Power (AC Mode)'] = date_filtered_data[(date_filtered_data['AC_Mode'] == 1) & (date_filtered_data['AC_and_DHW_Mode'] == 0)]['EP_OutdoorUnit_PowerSum_W'].mean()
    means['Outdoor Unit Power (Space Heat Mode)'] = date_filtered_data[(date_filtered_data['Space_Heat_Mode'] == 1) & (date_filtered_data['Defrost_Mode'] == 0)]['EP_OutdoorUnit_PowerSum_W'].mean()
    means['Outdoor Unit Power (Water Heating Mode)'] = date_filtered_data[
        (date_filtered_data['Water_Heating_Mode'] == 1) &
        (date_filtered_data['AC_and_DHW_Mode'] == 0) &
        (date_filtered_data['Defrost_Mode'] == 0) &
        (date_filtered_data['Space_Heat_Mode'] == 0)
    ]['EP_OutdoorUnit_PowerSum_W'].mean()
    
    means['Air Handler Power (AC Mode)'] = date_filtered_data[(date_filtered_data['AC_Mode'] == 1) & (date_filtered_data['AC_and_DHW_Mode'] == 0)]['EP_AH_PowerSum_W'].mean()
    means['Air Handler Power (Space Heat Mode)'] = date_filtered_data[(date_filtered_data['Space_Heat_Mode'] == 1) & (date_filtered_data['Defrost_Mode'] == 0)]['EP_AH_PowerSum_W'].mean()
    means['Air Handler Power (Water Heating Mode)'] = date_filtered_data[
        (date_filtered_data['Water_Heating_Mode'] == 1) &
        (date_filtered_data['AC_and_DHW_Mode'] == 0) &
        (date_filtered_data['Defrost_Mode'] == 0) &
        (date_filtered_data['Space_Heat_Mode'] == 0)
   
    ]['EP_AH_PowerSum_W'].mean()
    
    # Update figures with the latest mean annotations
    def add_latest_mean_annotation(fig, mean_value, label):
        if not pd.isna(mean_value):
            fig.add_annotation(
                x=0.01,
                y=0.80,  # Position below the yearly means
                xref='paper',
                yref='paper',
                text=f"Latest Mean: {mean_value:.2f} W",
                showarrow=False,
                font=dict(color="red", size=12),
                align='left'
            )
        return fig
    
    fig_ac_mode = create_static_histogram(ac_mode_data, 'EP_OutdoorUnit_PowerSum_W', 'Space Cooling Mode', 'Outdoor Unit Power Sum (W)')
    fig_ac_mode = add_latest_mean_annotation(fig_ac_mode, means['Outdoor Unit Power (AC Mode)'], 'Outdoor Unit Power (AC Mode)')
    
    fig_space_heat = create_static_histogram(space_heat_data, 'EP_OutdoorUnit_PowerSum_W', 'Space Heating Mode', 'Outdoor Unit Power Sum (W)')
    fig_space_heat = add_latest_mean_annotation(fig_space_heat, means['Outdoor Unit Power (Space Heat Mode)'], 'Outdoor Unit Power (Space Heat Mode)')
    
    fig_water_heating = create_static_histogram(water_heating_data, 'EP_OutdoorUnit_PowerSum_W', 'Water Heating Mode', 'Outdoor Unit Power Sum (W)')
    fig_water_heating = add_latest_mean_annotation(fig_water_heating, means['Outdoor Unit Power (Water Heating Mode)'], 'Outdoor Unit Power (Water Heating Mode)')
    
    fig_ah_power_ac = create_static_histogram(ah_power_ac_data, 'EP_AH_PowerSum_W', 'Space Cooling Mode', 'Air Handler Power Sum (W)')
    fig_ah_power_ac = add_latest_mean_annotation(fig_ah_power_ac, means['Air Handler Power (AC Mode)'], 'Air Handler Power (AC Mode)')
    
    fig_ah_power_space_heat = create_static_histogram(ah_power_space_heat_data, 'EP_AH_PowerSum_W', 'Space Heating mode', 'Air Handler Power Sum (W)')
    fig_ah_power_space_heat = add_latest_mean_annotation(fig_ah_power_space_heat, means['Air Handler Power (Space Heat Mode)'], 'Air Handler Power (Space Heat Mode)')
    
    fig_ah_power_water_heating = create_static_histogram(ah_power_water_heating_data, 'EP_AH_PowerSum_W', 'Water Heating Mode', 'Air Handler Power Sum (W)')
    fig_ah_power_water_heating = add_latest_mean_annotation(fig_ah_power_water_heating, means['Air Handler Power (Water Heating Mode)'], 'Air Handler Power (Water Heating Mode)')
    
    return fig_ac_mode, fig_space_heat, fig_water_heating, fig_ah_power_ac, fig_ah_power_space_heat, fig_ah_power_water_heating


if __name__ == '__main__':
    app.run_server(debug=True)