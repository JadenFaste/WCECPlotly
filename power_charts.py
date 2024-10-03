import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

# Initialize the Dash app
app = dash.Dash(__name__)
app.title = "HVAC Power Consumption Dashboard"

# Set latest date range
latest_begin = '2024-09-10'
latest_end = '2024-09-26'

# Define paths to your CSV files
path_5min_2022 = r"C:\Users\bober\OneDrive\Documents\GitHub\WCECPlotly\Villara 3 Function 5 Minute Data - 2022.csv"
path_5min_2023 = r"C:\Users\bober\OneDrive\Documents\GitHub\WCECPlotly\Villara 3 Function 5 Minute Data - 2023.csv"
path_5min_2024 = r"C:\Users\bober\OneDrive\Documents\GitHub\WCECPlotly\Villara 3 Function 5 Minute Data - 2024.csv"

path_cycle_2022 = r"C:\Users\bober\OneDrive\Documents\GitHub\WCECPlotly\Villara 3 Function Cycle Data - 2022.csv"
path_cycle_2023 = r"C:\Users\bober\OneDrive\Documents\GitHub\WCECPlotly\Villara 3 Function Cycle Data - 2023.csv"
path_cycle_2024 = r"C:\Users\bober\OneDrive\Documents\GitHub\WCECPlotly\Villara 3 Function Cycle Data - 2024.csv"

# Function to read and resample 5-minute data
def get_data(path):
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y %H:%M')
    df['Datetime'] = pd.to_datetime(df['Unnamed: 0'])
    df.set_index('Datetime', inplace=True)
    df = df.resample('5Min').mean()
    return df

# Load 5-minute data
df_5min_2022 = get_data(path_5min_2022)
df_5min_2023 = get_data(path_5min_2023)
df_5min_2024 = get_data(path_5min_2024)

# Load cycle data
def load_cycle_data(path):
    df = pd.read_csv(path)
    df['idx_Mid_Cycle'] = pd.to_datetime(df['idx_Mid_Cycle'])
    df.set_index('idx_Mid_Cycle', inplace=True)
    return df

df_cycle_2022 = load_cycle_data(path_cycle_2022)
df_cycle_2023 = load_cycle_data(path_cycle_2023)
df_cycle_2024 = load_cycle_data(path_cycle_2024)

# Helper function to create histograms
def create_histogram(df_2022, df_2023, df_2024, df_latest, column, title, xlabel, color_map):
    fig = go.Figure()

    # Add histograms for each year if data is available
    if not df_2022.empty:
        fig.add_trace(go.Histogram(
            x=df_2022[column],
            name='2022',
            marker_color=color_map['2022'],
            opacity=0.5
        ))
    if not df_2023.empty:
        fig.add_trace(go.Histogram(
            x=df_2023[column],
            name='2023',
            marker_color=color_map['2023'],
            opacity=0.5
        ))
    if not df_2024.empty:
        fig.add_trace(go.Histogram(
            x=df_2024[column],
            name='2024',
            marker_color=color_map['2024'],
            opacity=0.5
        ))

    # Update layout for overlapping histograms
    fig.update_layout(barmode='overlay', title=title, xaxis_title=xlabel, yaxis_title='Count')

    # Add mean lines and annotations
    min_xlim = min(df_2022[column].min() if not df_2022.empty else float('inf'),
                  df_2023[column].min() if not df_2023.empty else float('inf'),
                  df_2024[column].min() if not df_2024.empty else float('inf'))
    max_ylim = max(len(df_2022), len(df_2023), len(df_2024)) if not df_2022.empty or not df_2023.empty or not df_2024.empty else 1

    if not df_2022.empty:
        mean_2022 = df_2022[column].mean()
        fig.add_vline(x=mean_2022, line=dict(color=color_map['2022'], dash='dash'), annotation=dict(
            text=f"2022 Mean: {mean_2022:.2f}",
            x=mean_2022,
            y=max_ylim * 0.9,
            showarrow=False,
            xanchor='left',
            font=dict(color=color_map['2022'])
        ))
    if not df_2023.empty:
        mean_2023 = df_2023[column].mean()
        fig.add_vline(x=mean_2023, line=dict(color=color_map['2023'], dash='dash'), annotation=dict(
            text=f"2023 Mean: {mean_2023:.2f}",
            x=mean_2023,
            y=max_ylim * 0.8,
            showarrow=False,
            xanchor='left',
            font=dict(color=color_map['2023'])
        ))
    if not df_2024.empty:
        mean_2024 = df_2024[column].mean()
        fig.add_vline(x=mean_2024, line=dict(color=color_map['2024'], dash='dash'), annotation=dict(
            text=f"2024 Mean: {mean_2024:.2f}",
            x=mean_2024,
            y=max_ylim * 0.7,
            showarrow=False,
            xanchor='left',
            font=dict(color=color_map['2024'])
        ))

    # Add latest mean
    if not df_latest.empty:
        latest_mean = df_latest[column].mean()
        fig.add_vline(x=latest_mean, line=dict(color='black', dash='solid'), annotation=dict(
            text=f"Latest Mean: {latest_mean:.2f}",
            x=latest_mean,
            y=max_ylim * 0.6,
            showarrow=False,
            xanchor='left',
            font=dict(color='black')
        ))

    fig.update_layout(legend=dict(title='Year'))
    return fig

# Define color mapping for consistency
color_map = {
    '2022': 'blue',
    '2023': 'red',
    '2024': 'green'
}

# Function to filter and prepare data for ODU and AHU Power histograms
def prepare_data(mode_filters, additional_filters=None):
    """
    mode_filters: dict specifying which modes to filter on.
    additional_filters: list of tuples (column, value) to apply additional filtering.
    """
    df_latest = df_cycle_2024.loc[latest_begin:latest_end]

    # Apply mode filters
    for key, value in mode_filters.items():
        df_cycle_2022_filtered = df_cycle_2022[df_cycle_2022[key] == value]
        df_cycle_2023_filtered = df_cycle_2023[df_cycle_2023[key] == value]
        df_cycle_2024_filtered = df_cycle_2024[df_cycle_2024[key] == value]
        df_latest_filtered = df_latest[df_latest[key] == value]
    # Apply additional filters if any
    if additional_filters:
        for col, val in additional_filters:
            df_cycle_2022_filtered = df_cycle_2022_filtered[df_cycle_2022_filtered[col] == val]
            df_cycle_2023_filtered = df_cycle_2023_filtered[df_cycle_2023_filtered[col] == val]
            df_cycle_2024_filtered = df_cycle_2024_filtered[df_cycle_2024_filtered[col] == val]
            df_latest_filtered = df_latest_filtered[df_latest_filtered[col] == val]

    return df_cycle_2022_filtered, df_cycle_2023_filtered, df_cycle_2024_filtered, df_latest_filtered

# Function to prepare data for Water Heating modes with additional specific filters
def prepare_wh_data():
    df_latest = df_cycle_2024.loc[latest_begin:latest_end]

    # Apply Water Heating Mode
    df_cycle_2022_wh = df_cycle_2022[df_cycle_2022['Water_Heating_Mode'] == 1]
    df_cycle_2023_wh = df_cycle_2023[df_cycle_2023['Water_Heating_Mode'] == 1]
    df_cycle_2024_wh = df_cycle_2024[df_cycle_2024['Water_Heating_Mode'] == 1]
    df_latest_wh = df_latest[df_latest['Water_Heating_Mode'] == 1]

    # Apply additional filters
    additional_filters = [
        ('AC_and_DHW_Mode', 0),
        ('Defrost_Mode', 0),
        ('Space_Heat_Mode', 0)
    ]

    for col, val in additional_filters:
        df_cycle_2022_wh = df_cycle_2022_wh[df_cycle_2022_wh[col] == val]
        df_cycle_2023_wh = df_cycle_2023_wh[df_cycle_2023_wh[col] == val]
        df_cycle_2024_wh = df_cycle_2024_wh[df_cycle_2024_wh[col] == val]
        df_latest_wh = df_latest_wh[df_latest_wh[col] == val]

    # Drop cycles from 2022 and 2023 where ODU power < 400W
    df_cycle_2022_wh = df_cycle_2022_wh[df_cycle_2022_wh['EP_OutdoorUnit_PowerSum_W'] > 400]
    df_cycle_2023_wh = df_cycle_2023_wh[df_cycle_2023_wh['EP_OutdoorUnit_PowerSum_W'] > 400]

    return df_cycle_2022_wh, df_cycle_2023_wh, df_cycle_2024_wh, df_latest_wh

# Create all six figures
# 1. HVAC Cooling Mode ODU Power
def get_cool_mode_odu_power_fig():
    mode_filters = {'AC_Mode': 1}
    additional_filters = [('AC_and_DHW_Mode', 0)]
    df_2022, df_2023, df_2024, df_latest = prepare_data(mode_filters, additional_filters)
    return create_histogram(df_2022, df_2023, df_2024, df_latest,
                           column='EP_OutdoorUnit_PowerSum_W',
                           title='Outdoor Unit Power - Space Cooling Mode',
                           xlabel='Outdoor Unit Power [W]',
                           color_map=color_map)

# 2. HVAC Heating Mode ODU Power
def get_heat_mode_odu_power_fig():
    mode_filters = {'Space_Heat_Mode': 1}
    additional_filters = [('Defrost_Mode', 0)]
    df_2022, df_2023, df_2024, df_latest = prepare_data(mode_filters, additional_filters)
    return create_histogram(df_2022, df_2023, df_2024, df_latest,
                           column='EP_OutdoorUnit_PowerSum_W',
                           title='Outdoor Unit Power - Space Heating Mode',
                           xlabel='Outdoor Unit Power [W]',
                           color_map=color_map)

# 3. Water Heating Mode ODU Power
def get_wh_mode_odu_power_fig():
    df_2022_wh, df_2023_wh, df_2024_wh, df_latest_wh = prepare_wh_data()
    return create_histogram(df_2022_wh, df_2023_wh, df_2024_wh, df_latest_wh,
                           column='EP_OutdoorUnit_PowerSum_W',
                           title='Outdoor Unit Power - Water Heating Mode',
                           xlabel='Outdoor Unit Power [W]',
                           color_map=color_map)

# 4. HVAC Cooling Mode AHU Power
def get_cool_mode_ahu_power_fig():
    mode_filters = {'AC_Mode': 1}
    additional_filters = [('AC_and_DHW_Mode', 0)]
    df_2022, df_2023, df_2024, df_latest = prepare_data(mode_filters, additional_filters)
    return create_histogram(df_2022, df_2023, df_2024, df_latest,
                           column='EP_AH_PowerSum_W',
                           title='Air Handler Power - Space Cooling Mode',
                           xlabel='Air Handler Power [W]',
                           color_map=color_map)

# 5. HVAC Heating Mode AHU Power
def get_heat_mode_ahu_power_fig():
    mode_filters = {'Space_Heat_Mode': 1}
    additional_filters = [('Defrost_Mode', 0)]
    df_2022, df_2023, df_2024, df_latest = prepare_data(mode_filters, additional_filters)
    return create_histogram(df_2022, df_3, df_4, df_latest,
                           column='EP_AH_PowerSum_W',
                           title='Air Handler Power - Space Heating Mode',
                           xlabel='Air Handler Power [W]',
                           color_map=color_map)

# 6. Water Heating Mode AHU Power
def get_wh_mode_ahu_power_fig():
    df_2022_wh, df_2023_wh, df_2024_wh, df_latest_wh = prepare_wh_data()
    return create_histogram(df_2022_wh, df_2023_wh, df_2024_wh, df_latest_wh,
                           column='EP_AH_PowerSum_W',
                           title='Air Handler Power - Water Heating Mode',
                           xlabel='Air Handler Power [W]',
                           color_map=color_map)

# Generate all figures
fig_cool_odu = get_cool_mode_odu_power_fig()
fig_heat_odu = get_heat_mode_odu_power_fig()
fig_wh_odu = get_wh_mode_odu_power_fig()
fig_cool_ahu = get_cool_mode_ahu_power_fig()
fig_heat_ahu = get_heat_mode_ahu_power_fig()
fig_wh_ahu = get_wh_mode_ahu_power_fig()

# Define the layout of the Dash app
app.layout = html.Div([
    html.H1("HVAC Power Consumption Dashboard", style={'textAlign': 'center'}),
    html.Div([
        html.Div([
            dcc.Graph(
                id='cool-odu-power',
                figure=fig_cool_odu
            )
        ], className='six columns'),
        html.Div([
            dcc.Graph(
                id='heat-odu-power',
                figure=fig_heat_odu
            )
        ], className='six columns'),
    ], className='row'),
    html.Div([
        html.Div([
            dcc.Graph(
                id='wh-odu-power',
                figure=fig_wh_odu
            )
        ], className='six columns'),
        html.Div([
            dcc.Graph(
                id='cool-ahu-power',
                figure=fig_cool_ahu
            )
        ], className='six columns'),
    ], className='row'),
    html.Div([
        html.Div([
            dcc.Graph(
                id='heat-ahu-power',
                figure=fig_heat_ahu
            )
        ], className='six columns'),
        html.Div([
            dcc.Graph(
                id='wh-ahu-power',
                figure=fig_wh_ahu
            )
        ], className='six columns'),
    ], className='row'),
], style={'width': '95%', 'margin': 'auto'})

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
