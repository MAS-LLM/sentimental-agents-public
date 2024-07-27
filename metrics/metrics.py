import json
import os
import pandas as pd
import tempfile
from dotenv import load_dotenv
from pathlib import Path
from llama_hub.file.unstructured.base import UnstructuredReader
from llama_index.core import VectorStoreIndex, ServiceContext
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import torch

load_dotenv()

# Load resume samples
resume_df = pd.read_csv("data/data_utilities/output/data/resume_samples.csv")

device = "cuda" if torch.cuda.is_available() else "cpu"
embed_model = LangchainEmbedding(
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs={"device": device})
)

def load_resume(text: str) -> list:
    """
    Load resume data from text.

    Args:
    text (str): Resume text.

    Returns:
    list: List of loaded resume documents.
    """
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8") as tmpfile:
        tmpfile.write(text)
        tmp_path = tmpfile.name
        
    loader = UnstructuredReader()
    documents = loader.load_data(file=tmp_path)
    return documents

def run_defensibility_check(directory: str = 'output_files', export = False) -> list:
    """
    Run defensibility check on folders in a directory.

    Args:
    directory (str): Directory path.

    Returns:
    list: List of dataframes containing defensibility scores.
    """
    # Get all the folders in the directory
    folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]

    # Sort the folders based on modification time
    sorted_folders = sorted(folders, key=lambda f: os.path.getmtime(os.path.join(directory, f)))

    dfs_list = []
    sheet_names = []
    for i, folder_name in enumerate(sorted_folders):
        candidate_name = folder_name.split("_")[1]
        resume = resume_df.iloc[i]["resume"]
        with open(f"{directory}/{folder_name}/simulation_data.json", "r", encoding="utf-8") as jfile:
            data = json.load(jfile)

        documents = load_resume(resume)
        node_parser = SimpleNodeParser.from_defaults(chunk_size=64, chunk_overlap=32)
        service_context = ServiceContext.from_defaults(
            node_parser=node_parser,
            embed_model=embed_model)

        index = VectorStoreIndex.from_documents(documents, service_context=service_context)

        retriever = index.as_retriever()
        def_list = []
        for agent in data["agent_data"]:
            agent_name = agent["name"]
            for message in agent["messages"]:
                message_text = message["content"]
                response = retriever.retrieve(message_text)
                if len(response) == 0:
                    def_list.append({
                        "agent": agent_name,
                        "argument": message_text,
                        "source_text": "",
                        "score": 0
                    })
                else:
                    def_list.append({
                        "agent": agent_name,
                        "argument": message_text,
                        "source_text": response[0].node.text,
                        "score": response[0].score
                    })
        def_df = pd.DataFrame(def_list)
        dfs_list.append(def_df)
        sheet_names.append(candidate_name)
        print(candidate_name)
    if export:
        with pd.ExcelWriter(f"{directory}/defensibility_scores.xlsx", engine='openpyxl') as writer:
            for idx, df in enumerate(dfs_list):
                sheet_name = f'Sheet{idx+1}' if sheet_names is None else sheet_names[idx]
                df.to_excel(writer, sheet_name=sheet_name, index=False)
    return dfs_list

#run_defensibility_check()
