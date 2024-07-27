import json
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
import gensim
import gensim.corpora as corpora
from gensim.models import LdaModel
from gensim.corpora import Dictionary
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from gensim.utils import simple_preprocess
import pandas as pd

import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import os
import math


def load_simulation_data(directory):
    folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]

    # Sort the folders based on modification time
    sorted_folders = sorted(folders, key=lambda f: os.path.getmtime(os.path.join(directory, f)))
    sim_data = dict()
    for folder in sorted_folders:
        with open(f"{directory}/{folder}/simulation_data.json", "r", encoding="utf-8") as jfile:
            data = json.load(jfile)
        sim_data[folder.split("_")[-1]] = data  
    return sim_data




sim_data = load_simulation_data("output_files/exp101")

def prolificness_score(sim_data, directory):

    # Grid of plots
    num_candidates = len(sim_data)
    num_roles = 3  # Assuming two roles based on the simulated data
    rows, cols = math.ceil(len(sim_data) / 3), 3
    fig_width = cols * 4  # Adjust the multiplier as needed
    fig_height = rows * 3  # Adjust the multiplier as needed
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    
    
    for i, (candidate_name, candidate_data) in enumerate(sim_data.items()):
        unique_arguments_count_by_role = {}

        
        for entry in candidate_data["agent_data"]:
            # Splitting the text into individual statements for each role
            statements = [x["content"] for x in entry["messages"]]
            role = entry["name"]
            # Count the number of unique statements for each role
            unique_arguments_count_by_role[role] = len(statements)

        i_val = math.floor(i / 3)
        j_val = i % 3
        ax = axes[i_val, j_val]
        # Plotting
        #plt.figure(figsize=(10, 6))
        key = {
            'CFO':"CFO", 
            'VP of Engineering': "VE", 
            'Recycling Plant Manager':"RPM"
        }
        roles = [key[x] for x in unique_arguments_count_by_role.keys()]
        counts = list(unique_arguments_count_by_role.values())
        ax.bar(roles, counts)
        ax.set_xlabel('Role')
        ax.set_ylabel('Number of Unique Arguments')
        ax.set_title(f'Prolificness Score by Role for {candidate_name}')
        for k in range(len(roles)):
            ax.text(k, counts[k], counts[k], ha='center', va='bottom')
        #ax.set_xticks(rotation=45)
        #ax.set_xticks(rotation=360, fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{directory}/prolificness_score_by_role_grid.png")
    #return unique_arguments_count_by_role

dir_path = "output_files/exp101"

sim_data = load_simulation_data(dir_path)
prolificness_score(sim_data, dir_path)




def nuance_score(sim_data, directory):
    dfs = []
    candidate_names = []
    for candidate_name, candidate_data in sim_data.items():
        for entry in candidate_data["agent_data"]:
            # Splitting the text into individual statements for each role
            statements = [x["content"] for x in entry["messages"]]
        # Additional words to exclude along with stop words
        additional_exclude_words = set(candidate_name.split(" "))

        # Prepare the data
        stop_words = set(stopwords.words('english')).union(additional_exclude_words)
        data_words = [simple_preprocess(statement, deacc=True) for statement in
                    statements]  # Tokenize and remove punctuation

        # Remove Stop Words
        data_words_nostops = [[word for word in doc if word not in stop_words] for doc in data_words]

        # Create Dictionary and Corpus
        id2word = Dictionary(data_words_nostops)
        corpus = [id2word.doc2bow(text) for text in data_words_nostops]

        # Build LDA model
        lda_model = LdaModel(corpus=corpus,
                            id2word=id2word,
                            num_topics=5,  # Adjust the number of topics
                            random_state=100,
                            update_every=1,
                            chunksize=100,
                            passes=10,
                            alpha='auto')
        # Extract top words for each topic
        top_words_per_topic = dict()
        for i in range(lda_model.num_topics):
            top_words = [word for word, prob in lda_model.show_topic(i, 10)]  # Top 5 words
            top_words_per_topic[f"Topic {i}"] = top_words

        df = pd.DataFrame(top_words_per_topic)
        dfs.append(df)
        candidate_names.append(candidate_name)
        #for i in top_words_per_topic:
        #    print(i)

        ## Extracting the dominant topic for each statement
        #dominant_topics = []
        #for _, row_list in enumerate(lda_model[corpus]):
        #    row = sorted(row_list, key=lambda x: x[1], reverse=True)
        #    dominant_topics.append(row[0][0])

        # Counting the frequency of each dominant topic
        #topic_counts = pd.Series(dominant_topics).value_counts()

        ## Plotting
        #plt.figure(figsize=(10, 6))
        #plt.bar(x=topic_counts.index, height=topic_counts.values)
        #plt.xlabel('Topic')
        #plt.ylabel('Number of Statements')
        #plt.title('Distribution of Statements Across Topics')
        #plt.xticks(topic_counts.index, [f'Topic {i}' for i in topic_counts.index])
        #plt.savefig(f'plots/statement_topics_distribution _{candidate_name}.png')
        ## plt.show()
    ## write to excel
    writer = pd.ExcelWriter(f'{directory}/nuance_scores.xlsx')   
    for i, df in enumerate(dfs):
        df.to_excel(writer, sheet_name=candidate_names[i], index = False)
    writer._save()

dir_path = "output_files/exp21"
sim_data = load_simulation_data(dir_path)
nuance_score(sim_data, dir_path)


#data = sim_data['Melissa Morgan']
#agent = data["agent_data"][0]["name"]


def similarity_score(data, output_directory):
    # Organize messages by agent (role)
    messages_by_agent = defaultdict(list)
    for entry in data["agent_data"]:
        # Splitting the text into individual statements for each role
        agent_messages = [x["content"] for x in entry["messages"]]
        agent = entry["name"]
        messages_by_agent[agent].extend(agent_messages)

        

    # Function to compute cosine similarity
    def compute_cosine_similarity(messages):
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(messages)
        cosine_sim = cosine_similarity(tfidf_matrix)
        return cosine_sim

    # Compute intra-agent similarities
    intra_agent_similarities = {}
    for agent, messages in messages_by_agent.items():
        if messages:  # Checking if there are messages for the agent
            print(agent, messages)
            print("***********************************")
            intra_agent_similarities[agent] = compute_cosine_similarity(messages)

    # Revised code to compute inter-agent similarities by comparing each message after each new line.

    # Compute inter-agent similarities
    # Compute inter-agent similarities (revised approach)
    vectorizer = TfidfVectorizer()  # Vectorizer for inter-agent similarity
    agents = list(messages_by_agent.keys())
    inter_agent_similarities = np.zeros((len(agents), len(agents)))

    for i, agent1 in enumerate(agents):
        for j, agent2 in enumerate(agents):
            if i != j and messages_by_agent[agent1] and messages_by_agent[agent2]:
                similarities = []
                for message1 in messages_by_agent[agent1]:
                    for message2 in messages_by_agent[agent2]:
                        combined_tfidf = vectorizer.fit_transform([message1, message2])
                        cosine_sim = cosine_similarity(combined_tfidf)
                        similarities.append(cosine_sim[0, 1])
                inter_agent_similarities[i, j] = np.mean(similarities) if similarities else 0

    fig, axs = plt.subplots(1, len(intra_agent_similarities), figsize=(15, 5))
    for ax, (agent, similarity) in zip(axs, intra_agent_similarities.items()):
        cax = ax.matshow(similarity, cmap="coolwarm", vmin=0, vmax=1)
        ax.set_title(agent)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(f"{output_directory}/intra_agent_similarities.png", dpi=300)

    plt.figure(figsize=(8, 6))
    cax = plt.matshow(inter_agent_similarities, cmap="coolwarm", vmin=0, vmax=1)
    plt.title("Inter-Agent Similarities")
    # Change font size for x and y labels
    font_size = 6  # You can adjust this size as needed
    plt.xticks(range(len(agents)), agents, rotation=360, fontsize=font_size)
    plt.yticks(range(len(agents)), agents, fontsize=font_size, rotation=270)
    plt.colorbar(cax, fraction=0.046, pad=0.04)
    plt.savefig(f"{output_directory}/inter_agent_similarities.png", dpi=300)

dir_path = "output_files/exp26"
sim_data = load_simulation_data(dir_path)
for candidate_name,  candidate_data in sim_data.items():
    similarity_score(candidate_data, f"{dir_path}/candidate_{candidate_name}")





import torch
from polyfuzz import PolyFuzz
from polyfuzz.models import SentenceEmbeddings
from sentence_transformers import SentenceTransformer


device = "cuda" if torch.cuda.is_available() else "cpu"
embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
distance_model = SentenceEmbeddings(embedding_model)
model = PolyFuzz(distance_model)

dir_path = "output_files/exp24"
sim_data = load_simulation_data(dir_path)

def calculate_drift(sim_data, directory):
    drift_list = []
    for candidate_name, candidate_data in sim_data.items():
        agent_drift = {"candidate":candidate_name}
        for entry in candidate_data["agent_data"]:
            messages = [x["content"] for x in entry["messages"]]
            if len(messages) == 0:
                agent_drift[agent] = np.nan
            else:
                agent = entry["name"]
                system_messages = data["initial_conditions"]["agent_system_messages"][agent]
                df = model.match(messages, [system_messages]).get_matches()
                agent_drift[agent] = df["Similarity"].mean()
        drift_list.append(agent_drift)

    drift_df = pd.DataFrame(drift_list)
    drift_df.to_csv(f"{directory}/drift_df.csv", index = False)


calculate_drift(sim_data, dir_path)