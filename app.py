import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import pandas as pd
import dash_bootstrap_components as dbc
import plotly.express as px
import base64
import io
import random
from runner import run_simulation


app = dash.Dash(
    name="SentimentalAgents",
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME]
)

default_csv_path = 'data/input/resume_samples.csv'

app.layout = dbc.Container([
    html.H1("Resume Evaluation Dashboard", className="text-center my-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3("Job Title", className="card-title"),
                    dbc.Input(id='job-title', type='text', placeholder='Enter job title', className="mb-3"),
                ])
            ], className="h-100")
        ], width=12, lg=2),                    
        dbc.Col([
            dbc.Card([
                dbc.CardBody([                         
                    html.H3("Job Description", className="card-title"),
                    dbc.Textarea(id='job-description', placeholder='Enter job description...', 
                                 style={'height': '100px'}, className="mb-3"),
                ])
            ], className="h-100")
        ], width=12, lg=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([                                 
                    html.H3("Upload Resumes", className="card-title"),
                    dcc.Upload(
                        id='upload-resumes',
                        children=html.Div([
                            html.I(className="fas fa-file-csv fa-2x me-2"),
                            'Drag and Drop or ',
                            html.A('Select CSV File')
                        ]),
                        className="border rounded p-3 mb-3",
                    ),
                    html.Div(id='upload-status'),
                ])
            ], className="h-100")
        ], width=12, lg=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([                         
                    html.H3("Advisors", className="card-title"),
                    dbc.Textarea(id='advisors', placeholder='Enter list of advisors...', 
                                 style={'height': '100px'}, className="mb-3"),
                ])
            ], className="h-100")
        ], width=12, lg=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([                          
                    html.H3("Non-Bayesian Updating", className="card-title"),
                    dbc.Checklist(
                        id='non-bayesian',
                        options=[{'label': 'Enable Non-Bayesian Updating', 'value': 'enable'}],
                        value=[],
                        switch=True,
                        className="mb-3"
                    ),
                    html.Div(id='non-bayesian-params', children=[
                        dbc.Label("Tolerance"),
                        dbc.Input(id='tolerance', type='number', placeholder='Enter tolerance', className="mb-2"),
                        dbc.Label("Alpha"),
                        dbc.Input(id='alpha', type='number', placeholder='Enter alpha', className="mb-3")
                    ], style={'display': 'none'}),
                ])
            ], className="h-100")
        ], width=12, lg=2)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Button('Get agent profiles', id='process-button', color="primary", className="mt-3 mb-4")
        ], width=12)
    ]),
    
    html.Div(id='agent-tabs')
], fluid=True)

@app.callback(
    Output('non-bayesian-params', 'style'),
    Input('non-bayesian', 'value')
)
def toggle_non_bayesian_params(non_bayesian):
    if 'enable' in non_bayesian:
        return {'display': 'block'}
    return {'display': 'none'}

@app.callback(
    Output('upload-status', 'children'),
    Input('upload-resumes', 'contents'),
    State('upload-resumes', 'filename')
)
def update_upload_status(contents, filename):
    if contents is not None:
        return html.Div(f"Uploaded: {filename}")
    return html.Div("No file uploaded. Using default file.")

@app.callback(
    Output('agent-tabs', 'children'),
    Input('process-button', 'n_clicks'),
    State('job-title', 'value'),
    State('job-description', 'value'),
    State('upload-resumes', 'contents'),
    State('advisors', 'value'),
    State('non-bayesian', 'value'),
    State('tolerance', 'value'),
    State('alpha', 'value')
)
def process_data(n_clicks, job_title, job_description, resume_contents, advisors, non_bayesian, tolerance, alpha):
    if n_clicks is None:
        return []

    # Load CSV data (either from upload or default file)
    if resume_contents:
        content_type, content_string = resume_contents.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    else:
        df = pd.read_csv(default_csv_path)

    if 'enable' in non_bayesian:
        config = {'tolerance': tolerance, 'alpha': alpha}
    else:
        config = None
    # Generate tabs for each agent
    tabs = []
    for key, entry in :
        agent_name = row['Name']
        
        # Generate dummy data for demonstration
        criteria = f"Criteria for {agent_name}"
        priorities = f"Priorities for {agent_name}"
        objectives = f"Objectives for {agent_name}"
        topic = f"Discussion topic for {agent_name}"
        discussion = f"Sample discussion for {agent_name}"
        
        # Generate dummy sentiment data
        sentiment_data = pd.DataFrame({
            'Time': range(10),
            'Sentiment': [random.uniform(-1, 1) for _ in range(10)]
        })
        
        sentiment_fig = px.line(sentiment_data, x='Time', y='Sentiment', title=f'Sentiment Change for {agent_name}')
        
        tab_content = dbc.Card(
            dbc.CardBody([
                html.H4(f"Agent: {agent_name}"),
                html.P(f"Criteria: {criteria}"),
                html.P(f"Priorities: {priorities}"),
                html.P(f"Objectives: {objectives}"),
                html.H5("Topic of Discussion"),
                html.P(topic),
                html.H5("Agent Discussion"),
                html.P(discussion),
                dcc.Graph(figure=sentiment_fig)
            ])
        )
        
        tabs.append(dbc.Tab(tab_content, label=agent_name))

    return dbc.Tabs(tabs)

if __name__ == '__main__':
    app.run_server(debug=True)