import dash
from dash import dcc, html
import plotly.express as px
from dash.dependencies import Input, Output
import pandas as pd
import os

# Set the working directory
path = r'C:\Users\bober\OneDrive\Documents\GitHub\WCECPlotly'
os.chdir(path)

# Load the dataset
data = pd.read_csv('cycle_data_updated.csv')

# Convert 'idx_Mid_Cycle' to datetime
data['idx_Mid_Cycle'] = pd.to_datetime(data['idx_Mid_Cycle'], errors='coerce')
data['Year'] = data['idx_Mid_Cycle'].dt.year

# Filter the data where AC_Mode is 1, AC_and_DHW_Mode is 0, and COP_Steady_State > 0.2
filtered_df = data[(data['AC_Mode'] == 1) & (data['AC_and_DHW_Mode'] == 0) & (data['COP_Steady_State'] > 0.2)]

# Convert 'Year' to string to ensure it is treated as a categorical variable
filtered_df['Year'] = filtered_df['Year'].astype(str)

# Filter necessary columns and drop NaN
filtered_df = filtered_df[['T_Outdoor_ecobee_F', 'COP_Steady_State', 'idx_Mid_Cycle', 'Year']].dropna()

app = dash.Dash(__name__)

# App layout
app.layout = html.Div([
    html.H1("COP Steady State vs Outdoor Temperature by Year"),
    
    # Add a date picker
    dcc.DatePickerRange(
        id='date-picker',
        min_date_allowed=filtered_df['idx_Mid_Cycle'].min().date(),
        max_date_allowed=filtered_df['idx_Mid_Cycle'].max().date(),
        start_date='2024-07-21',  # Set default start date
        end_date='2024-07-22' 
    ),
    
    dcc.Graph(id='scatter-plot')
])

# Define callback to update the graph
@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date')]
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

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
