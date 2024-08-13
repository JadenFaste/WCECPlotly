import pandas as pd
import plotly.express as px
from datetime import datetime

# Modify your DataFrame to include a group column
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

# Create the Gantt chart using Plotly Express with groups
fig = px.timeline(tasks, x_start="Start", x_end="End", y="Task", color="Group", title="Project Schedule Gantt Chart")

# Update layout for better readability
fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Tasks",
    yaxis=dict(autorange="reversed"),
    title_x=0.5
)

# Add a vertical line for the current date
current_date = datetime.now().strftime('%Y-%m-%d')
fig.add_shape(
    type="line",
    x0=current_date,
    y0=0,
    x1=current_date,
    y1=1,
    xref='x',
    yref='paper',
    line=dict(color="Red", width=2, dash="dash")
)

# Show the chart
fig.show()