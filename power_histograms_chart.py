import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
import os

path = r'C:\Users\bober\OneDrive\Documents\GitHub\WCECPlotly'
os.chdir(path)

# Load and preprocess the data
data = pd.read_csv('cycle_data_updated.csv')
data['idx_Cycle_Start'] = pd.to_datetime(data['idx_Cycle_Start'], errors='coerce')
data['Year'] = data['idx_Cycle_Start'].dt.year

# Function to create static histograms with yearly means and mean lines
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

# Initialize the Dash app
app = Dash(__name__)

# Define layout with date picker and static plots in a grid
app.layout = html.Div([
    dcc.DatePickerRange(
        id='date-picker-range',
        min_date_allowed=data['idx_Cycle_Start'].min().date(),
        max_date_allowed=data['idx_Cycle_Start'].max().date(),
        start_date=data['idx_Cycle_Start'].min().date(),
        end_date=data['idx_Cycle_Start'].max().date()
    ),
    html.Div([
        dcc.Graph(id='ac-mode-plot', figure=fig_ac_mode, style={'display': 'inline-block', 'width': '32%'}),
        dcc.Graph(id='space-heat-plot', figure=fig_space_heat, style={'display': 'inline-block', 'width': '32%'}),
        dcc.Graph(id='water-heating-plot', figure=fig_water_heating, style={'display': 'inline-block', 'width': '32%'}),
    ], style={'display': 'flex', 'justify-content': 'space-between'}),
    html.Div([
        dcc.Graph(id='ah-power-ac-plot', figure=fig_ah_power_ac, style={'display': 'inline-block', 'width': '32%'}),
        dcc.Graph(id='ah-power-space-heat-plot', figure=fig_ah_power_space_heat, style={'display': 'inline-block', 'width': '32%'}),
        dcc.Graph(id='ah-power-water-heating-plot', figure=fig_ah_power_water_heating, style={'display': 'inline-block', 'width': '32%'}),
    ], style={'display': 'flex', 'justify-content': 'space-between'}),
])

# Define callback to update the figures with the latest mean based on date range selection
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

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)