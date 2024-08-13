import pandas as pd
import plotly.express as px
import openpyxl

# Load your Excel file
file_path = 'gantt.xlsx'
xls = pd.ExcelFile(file_path)
df_clean = pd.read_excel(xls, sheet_name='Project schedule', skiprows=4)

# Extract relevant columns: Task, Start, End
tasks_df = df_clean[['TASK', 'START', 'END']].dropna().reset_index(drop=True)
tasks_df.columns = ['Task', 'Start', 'End']

# Create the Gantt chart using Plotly Express
fig = px.timeline(tasks_df, x_start="Start", x_end="End", y="Task", title="Project Schedule Gantt Chart")
fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Tasks",
    yaxis=dict(autorange="reversed"),
    title_x=0.5
)

# Show the chart
fig.show()
