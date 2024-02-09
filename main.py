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
from tabulate import tabulate


# Load the CSV data
file_path = "C:/Datasets/WCEC Datasets/WCEC dataspell environment/Test data.csv"
filepath_2 = "C:/Datasets/WCEC Datasets/WCEC dataspell environment/Villara 3 Function Cycle Data.csv"
df = pd.read_csv(file_path)
df_2 = pd.read_csv(filepath_2)
# df_2 =df_2[~df.isin([np.nan, np.inf, -np.inf]).any(1)]

# Declare variables for regression
x = df_2["T_Outdoor_ecobee_F"]
y = df_2["COP"]
x = pd.to_numeric(x, errors='coerce')
y = pd.to_numeric(y, errors='coerce')
x = np.nan_to_num(x, nan=0, posinf=0, neginf = 0)
y = np.nan_to_num(y, nan=0, posinf = 0, neginf = 0)

binary_columns = ['AC_Mode', 'AC_and_DHW_Mode', 'Space_Heat_Mode', 'Water_Heating_Mode', 'Defrost_Mode']

# Convert the 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y %H:%M')

# Avg of t2 and t3
df['T_HotTank_T2_T3_avg_F'] = (df['T_HotTank_T2_F'] + df['T_HotTank_T3_F']) / 2

# Variables available for plotting (excluding the Date column for the dropdown)
available_variables = df.columns.drop('Date')


mode_colors = {
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
                        fillcolor=mode_colors[col],  # Use the color from our dictionary
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
                    fillcolor=mode_colors[col],  # Use the color from our dictionary
                    opacity=0.5,
                    layer="below",
                    line_width=0,
                )
            )
        all_shapes[col] = shapes
    return all_shapes

# Create the Dash app
app = dash.Dash(__name__)
# Define the layout
app.layout = html.Div([
    html.H1("Time Series Dashboard"),

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

    # DatePickerRange to filter by date range
    html.Div([
        html.Label("Select Date Range:"),
        dcc.DatePickerRange(
            id='date-picker-range',
            start_date=df['Date'].min(),
            end_date=df['Date'].max(),
            display_format='MM/DD/YYYY'
        )
    ], style={'margin-top': '20px', 'margin-bottom': '20px'}),

    # Checkboxes for binary columns
    html.Div([
        html.Label("Select Modes to Highlight:"),
        dcc.Checklist(
            id='mode-selector',
            options=[
                {'label': 'AC and DHW Mode', 'value': 'Controller_AC_and_DHW_Mode'},
                {'label': 'Defrost Mode', 'value': 'Controller_Defrost_Mode'},
                {'label': 'Space Heat Mode', 'value': 'Controller_Space_Heat_Mode'},
                {'label': 'Water Heating Mode', 'value': 'Controller_Water_Heating_Mode'}
            ],
            value=['Controller_Water_Heating_Mode'],  # Default value
            inline=True
        ),
    ], style={'margin-top': '20px', 'margin-bottom': '20px'}),

    # Graph for displaying the line chart
    dcc.Graph(id='line-plot'),

    html.Div(id='equation'),  # Div to display the equation

    # Separator
    html.Hr(),

    # Second Graph
    dcc.Graph(id='fixed-variables-plot'),

    # Separator
    html.Hr(),

    # Third Graph
    dcc.Graph(id='custom-variables-plot'),

    # Seperator
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
    html.Div(id='equation')  # Div to display the equation
])

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
    # fig.add_traces(go.Scatter(x=reg_df['x'], y=reg_df['model'], mode='lines', name='model'))
    fig.add_traces(go.Scatter(x=reg_df['x'], y=reg_df['model'], mode='lines', name='model'))


    fig.update_layout(
        title="Polynomial Regression Analysis",
        xaxis_title="Outdoor Temperature (F)",
        yaxis_title="Coefficient of Performance (COP)"
    )

    # Return the figure and the regression equation
    return fig, equation_and_metrics

#####################
@app.callback(
    [Output('line-plot', 'figure'),
     Output('fixed-variables-plot', 'figure'),
     Output('custom-variables-plot', 'figure')],
    [Input('primary-yaxis-column-name', 'value'),
     Input('secondary-yaxis-column-name', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date'),
     Input('mode-selector', 'value')]
)
def update_graph(primary_var, secondary_var, start_date, end_date, modes):
    filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

    fig1 = px.line(filtered_df, x='Date', y=primary_var)
    fig1.add_traces(go.Scatter(x=filtered_df['Date'], y=filtered_df[secondary_var],
                               mode='lines', name=secondary_var))
    all_shapes = get_all_shaded_regions(filtered_df, modes)
    for mode, shapes in all_shapes.items():
        for shape in shapes:
            fig1.add_shape(shape)
    fig1.update_layout(
        title=f'Time Series of {primary_var} & {secondary_var}',
        legend=dict(
            orientation='h',  # Horizontal orientation
            yanchor='bottom',  # Anchor legend to the bottom
            y=-0.5,  # Position legend below the x-axis
            xanchor='center',  # Center the legend horizontally
            x=0.5  # Position the center of the legend in the middle of the plot
        )
    )


    fig2 = go.Figure()

    yaxes = ['y', 'y2', 'y3', 'y4']
    colors = ['blue', 'red', 'green', 'purple']

    fixed_variables = ["EP_Total_HVAC_Power_W", "T_Outdoor_ecobee_F", "VFR_HotTank_WaterDraw_FlowRate_gpm",
                       "T_HotTank_T2_T3_avg_F"]

    for i, var in enumerate(fixed_variables):
        fig2.add_traces(go.Scatter(x=filtered_df['Date'], y=filtered_df[var],
                                   mode='lines', name=var, yaxis=yaxes[i], line=dict(color=colors[i])))

    for mode, shapes in all_shapes.items():
        for shape in shapes:
            fig2.add_shape(shape)

    fig2.update_layout(
        yaxis=dict(title=fixed_variables[0], side="right", position=.95),
        yaxis2=dict(title=fixed_variables[1], overlaying='y', side='left', anchor="free", position=.05),
        yaxis3=dict(title=fixed_variables[2], overlaying='y', side='right', anchor="free", position=1),
        yaxis4=dict(title=fixed_variables[3], overlaying='y', side='left', anchor="free", position=0.15),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.5,  # Negative values to place the legend below the plot
            xanchor='center',
            x=0.5
        )
    )

    fig3 = go.Figure()

    yaxes = ['y', 'y2', 'y3', 'y4', 'y5', 'y6']
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'pink']

    custom_variables = ["EP_Total_HVAC_Power_W", "T_Outdoor_ecobee_F", "T_CoolSetpoint_F", "T_HeatSetpoint_F",
                        "T_Thermostat_F", "T_HeatSetpoint_F"]

    for i, var in enumerate(custom_variables):
        fig3.add_traces(go.Scatter(x=filtered_df['Date'], y=filtered_df[var],
                                   mode='lines', name=var, yaxis=yaxes[i], line=dict(color=colors[i])))

    for mode, shapes in all_shapes.items():
        for shape in shapes:
            fig3.add_shape(shape)

    fig3.update_layout(
        yaxis=dict(title=custom_variables[0], side='right'),
        yaxis2=dict(title=custom_variables[1], overlaying='y', side='left', anchor="free", position = 0),
        yaxis3=dict(title=custom_variables[2], overlaying='y', side='left', anchor="free", position= 0.05),
        yaxis4=dict(title=custom_variables[3], overlaying='y', side='left', anchor="free", position=0.1),
        yaxis5=dict(title=custom_variables[4], overlaying='y', side='left', anchor="free", position=0.15),
        yaxis6=dict(title=custom_variables[5], overlaying='y', side='left', anchor="free", position=0.2),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.5,  # Negative values to place the legend below the plot
            xanchor='center',
            x=0.5
        )
    )
    title='Time Series of Custom Variables'

    return fig1, fig2, fig3

if __name__ == '__main__':
    app.run_server(debug=True)
