import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_change_sentiment_plot(non_bayesian_data):
    # Data
    roles = list(non_bayesian_data['change'].keys())

    # Find the maximum length of data across all roles
    max_length = max(max(len(non_bayesian_data['change'][role]), len(non_bayesian_data['sentiment_data'][role])) for role in roles)

    # Define a color for each role
    colors = {'CFO': 'blue', 'VP of Engineering': 'red', 'Recycling Plant Manager': 'green'}

    # Create subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("Change Over Time", "Sentiment Over Time"))

    # Add traces for change data
    for role in roles:
        x = list(range(len(non_bayesian_data['change'][role])))
        fig.add_trace(
            go.Scatter(x=x, y=non_bayesian_data['change'][role], name=f"{role} - Change", 
                       mode='lines+markers', line=dict(color=colors[role]), 
                       marker=dict(color=colors[role])),
            row=1, col=1
        )

    # Add traces for sentiment data
    for role in roles:
        x = list(range(len(non_bayesian_data['sentiment_data'][role])))
        fig.add_trace(
            go.Scatter(x=x, y=non_bayesian_data['sentiment_data'][role], name=f"{role} - Sentiment", 
                       mode='lines+markers', line=dict(color=colors[role]), 
                       marker=dict(color=colors[role])),
            row=2, col=1
        )

    # Update layout
    fig.update_layout(
        height=800,
        title_text="Change and Sentiment Over Time for Different Roles",
        legend_title_text="Role and Metric",
        xaxis_title="Conversation Round",
    )

    fig.update_xaxes(range=[-0.5, max_length - 0.5])  # Set x-axis range to accommodate all data points
    fig.update_yaxes(title_text="Change", row=1, col=1)
    fig.update_yaxes(title_text="Sentiment", row=2, col=1)

    return fig

# Usage example:
# fig = create_change_sentiment_plot(non_bayesian_data)
# fig.show()