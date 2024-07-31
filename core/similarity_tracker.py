from sentence_transformers import SentenceTransformer
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the model
model = SentenceTransformer("BAAI/bge-large-en-v1.5")
model = model.to(device)

with open("output_files/20240731_150705/candidate_David Bishop/simulation_data.json", "r") as jfile:
    test_data = json.load(jfile)["raw_history"]






class SentimentTracker:
    def __init__(self, agents: list, visualize = True) -> None:
        self.visualize = visualize
        self.similarities = {agent: [] for agent in agents}
        self.messages = {agent: [] for agent in agents}
        self.text_embeddings = []

    
    def _get_embedding(self, text):
        if device == "cuda":
            return model.encode(text).cpu().numpy()
        else:
            return model.encode(text)
    
    def update(self, name, message):
        posterior = self._get_embedding(message)
        self.text_embeddings.append(posterior)
        if len(self.messages[name]) > 0:
            prior = self._get_embedding(self.messages[name][-1])
            similarity = cosine_similarity([prior], [posterior])[0][0]
            self.similarities[name].append(similarity)
        self.messages[name].append(message)
            

    def visualize(self):
        pass

agents = []
data = []
for row in test_data:
    name, message = row.split("): ")
    name = name.strip("(")
    agents.append(name)
    data.append((name, message))

agents = list(set(agents))


tracker = SentimentTracker(agents)

for name, message in data:
    tracker.update(name, message)


from pprint import pprint as pp
pp(tracker.similarities)



import plotly.graph_objects as go
from scipy import interpolate

# Create figure
fig = go.Figure()

# Add traces for each role
for role, values in tracker.similarities.items():
    fig.add_trace(go.Scatter(
        x=list(range(1, len(values) + 1)),
        y=values,
        mode='lines+markers',
        name=role
    ))

# Update layout
fig.update_layout(
    title='Similarity change by Role Over Time',
    xaxis_title='Change in Round',
    yaxis_title='Performance Score',
    legend_title='Role',
    hovermode='x unified'
)

fig.write_html("plot_outputs/similarity_change.html")

fig.show()

len(tracker.text_embeddings)
import numpy as np
from umap import UMAP

reducer = UMAP(n_components = 3)

embeddings = np.array(tracker.text_embeddings)
embeddings.shape

r_embeddings = reducer.fit_transform(embeddings)


colors = []
text_list = []
for i in range(0, len(tracker.text_embeddings), 3):
    c_round = i//3 + 1
    for agent, color in zip(agents, ["#1E90FF", "#FF69B4", "#2ECC71"]):
        message = tracker.messages[agent][i//3]
        colors.append(color)
        text_list.append(
            f"{agent} in Round {c_round} <br> <br> {message[:80]} <br> {message[80:160]} <br> {message[160:240]}"
        )

# Create the 3D scatter plot
fig = go.Figure()

# Add traces for each agent
for agent, color in zip(agents, ["#1E90FF", "#FF69B4", "#2ECC71"]):
    agent_indices = [i for i in range(len(colors)) if colors[i] == color]
    fig.add_trace(go.Scatter3d(
        x=r_embeddings[agent_indices, 0],
        y=r_embeddings[agent_indices, 1],
        z=r_embeddings[agent_indices, 2],
        mode='markers',
        marker=dict(
            size=5,
            color=color,
            opacity=0.8
        ),
        text=[text_list[i] for i in agent_indices],
        hoverinfo='text',
        name=agent  # This will appear in the legend
    ))

# Update the layout
fig.update_layout(
    title='3D UMAP Agent Message Embeddings',
    scene=dict(
        xaxis_title='UMAP 1',
        yaxis_title='UMAP 2',
        zaxis_title='UMAP 3'
    ),
    width=900,
    height=700,
    margin=dict(r=20, b=10, l=10, t=40),
    legend_title_text='Agents',
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    )
)


fig.write_html("plot_outputs/embedding_plot.html")

# Show the plot
fig.show()