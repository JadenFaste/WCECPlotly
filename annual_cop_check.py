import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import os

path = r'C:\Users\bober\OneDrive\Documents\GitHub\WCECPlotly'
os.chdir(path)

# Load the updated dataset
cycle_data_updated = pd.read_csv('cycle_data_updated.csv')

# Convert index to datetime
cycle_data_updated['idx_Mid_Cycle'] = pd.to_datetime(cycle_data_updated['idx_Mid_Cycle'])

def ac_mode_cop_plot_plotly(cycle_data_updated, latest_begin, latest_end):
    # Filter data
    filtered_data = cycle_data_updated[
        (cycle_data_updated['AC_Mode'] == 1) & 
        (cycle_data_updated['AC_and_DHW_Mode'] == 0)
    ]
    
    filtered_data_2022 = filtered_data[filtered_data['idx_Mid_Cycle'].dt.year == 2022]
    filtered_data_2023 = filtered_data[filtered_data['idx_Mid_Cycle'].dt.year == 2023]
    filtered_data_2024 = filtered_data[filtered_data['idx_Mid_Cycle'].dt.year == 2024]
    
    filtered_data_2024 = filtered_data_2024[filtered_data_2024['idx_Mid_Cycle'] >= '2024-07-19']
    filtered_data_latest = filtered_data_2024[
        (filtered_data_2024['idx_Mid_Cycle'] >= latest_begin) & 
        (filtered_data_2024['idx_Mid_Cycle'] <= latest_end)
    ]
    
    # Create Plotly figure
    fig = go.Figure()

    # Add scatter traces for each year
    fig.add_trace(go.Scatter(
        x=filtered_data_2022['T_Outdoor_ecobee_F'],
        y=filtered_data_2022['COP_Steady_State'],
        mode='markers',
        name='2022',
        marker=dict(opacity=0.5)
    ))

    fig.add_trace(go.Scatter(
        x=filtered_data_2023['T_Outdoor_ecobee_F'],
        y=filtered_data_2023['COP_Steady_State'],
        mode='markers',
        name='2023',
        marker=dict(opacity=0.5)
    ))

    fig.add_trace(go.Scatter(
        x=filtered_data_2024['T_Outdoor_ecobee_F'],
        y=filtered_data_2024['COP_Steady_State'],
        mode='markers',
        name='2024',
        marker=dict(opacity=0.5)
    ))

    # Current Selection
    fig.add_trace(go.Scatter(
        x=filtered_data_latest['T_Outdoor_ecobee_F'],
        y=filtered_data_latest['COP_Steady_State'],
        mode='markers',
        name='Current Selection',
        marker=dict(color='black', size=10, line=dict(width=2, color='DarkSlateGrey'))
    ))

    # Update layout
    fig.update_layout(
        title='AC Mode',
        xaxis_title='Outdoor Air Temperature [°F]',
        yaxis_title='Steady State COP',
        yaxis=dict(range=[0, 6]),
        legend=dict(title='Year'),
        template='plotly_white',
        hovermode='closest'
    )

    return fig


# Initialize Dash app
app = dash.Dash(__name__)
app.title = "AC Mode COP Plot"

# Define the layout
app.layout = html.Div([
    html.H1("AC Mode COP Analysis"),
    
    # Date Picker for selecting the latest_begin and latest_end
    html.Div([
        html.Label("Select Date Range for Current Selection:"),
        dcc.DatePickerRange(
            id='date-picker-range',
            min_date_allowed=cycle_data_updated['idx_Mid_Cycle'].min().date(),
            max_date_allowed=cycle_data_updated['idx_Mid_Cycle'].max().date(),
            start_date='2024-08-10',
            end_date='2024-08-11'
        )
    ], style={'margin': '20px'}),
    
    # Placeholder for the Plotly graph
    dcc.Graph(
        id='cop-plot'
    )
])

# Define callback to update the plot based on selected dates
@app.callback(
    Output('cop-plot', 'figure'),
    [Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_cop_plot(start_date, end_date):
    # Generate the plotly figure with the selected dates
    fig = ac_mode_cop_plot_plotly(cycle_data_updated, start_date, end_date)
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)