import os
import pandas as pd
import nltk
import torch
from typing import List, Dict, Callable, Tuple, Any, Union
from polyfuzz import PolyFuzz
from polyfuzz.models import SentenceEmbeddings
from sentence_transformers import SentenceTransformer
from llama_index.llms.openai import OpenAI
from llama_index.program.openai import OpenAIPydanticProgram
from pydantic import BaseModel, ValidationError
from core.advisory_brief import ANALYTICS_TEMPLATE
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class StrongOpinions(BaseModel):
    """
    Data model for strong opinions. These represent viewpoints or beliefs that are held with firm conviction and expressed with confidence and certainty.
    """
    strong_opinions: List[str]

class AgreeableOpinions(BaseModel):
    """
    Data model for agreeable opinions. These represent viewpoints that are generally accepted or approved by most people in a conversation or discussion.
    """
    agreeable_opinions: List[str]

class ExtensivelyDiscussedOpinions(BaseModel):
    """
    Data model for extensively discussed opinions. These represent viewpoints or beliefs that have been thoroughly examined, debated, or talked about in a conversation or discussion.
    """
    extensively_discussed_opinions: List[str]

class NegativeOpinions(BaseModel):
    """
    Data model for negative opinions. These represent viewpoints or beliefs that are held with a strong suspicion or doubt.
    """
    negative_opinions: List[str]


class Report(BaseModel):
    """
    Data model for a candidate report. This report includes strong opinions, agreeable opinions, and extensively discussed opinions.
    """
    strong_opinions: StrongOpinions
    agreeable_opinions: AgreeableOpinions
    extensively_discussed_opinions: ExtensivelyDiscussedOpinions
    negative_opinions: NegativeOpinions

    def generate_report(self) -> Dict[str, List[str]]:
        """
        Generate a dictionary representation of the report.
        """
        return {
            "Strong Opinions": self.strong_opinions.strong_opinions,
            "Agreeable Opinions": self.agreeable_opinions.agreeable_opinions,
            "Extensively Discussed Opinions": self.extensively_discussed_opinions.extensively_discussed_opinions,
            "Negative Opinions": self.negative_opinions.negative_opinions,
        }

    def pretty_print(self) -> None:
        """
        Pretty print the report.
        """
        report = self.generate_report()
        for key, value in report.items():
            print(f"{key}:\n{'-'*len(key)}")
            for opinion in value:
                print(f"- {opinion}")
            print("\n")




class AdvisorReport:
    def __init__(self, agents: list, rounds: int = 100):
        self.agents = agents
        self.rounds = rounds
        self.OPENAI_MODEL = os.getenv("OPENAI_MODEL")
        self.llm = self.get_llm(self.OPENAI_MODEL)
        self.program = self.get_program(self.llm, Report, ANALYTICS_TEMPLATE)
        self.advisor_messages, self.cdf = self.get_advisor_messages(self.agents)
        self.report = self.get_report(self.program, self.advisor_messages)
        self.report_dict = self.report.generate_report()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)
        self.distance_model = SentenceEmbeddings(self.embedding_model)
        self.model = PolyFuzz(self.distance_model)


    def get_advisor_messages(self, agents: list, rounds: int = 100) -> str:
        """
        This function generates the messages of each advisor in a round-robin fashion.

        Parameters:
        agents (list): A list of agent objects. Each agent object should have 'name' and 'own_messages' attributes.
        rounds (int): The number of rounds to generate messages for. Default is 100.

        Returns:
        str: A string containing all the advisor messages.
        """
        # Initialize the conversation round
        convo_round = 0

        # Initialize the output list
        out = []
        df_data = []

        # Loop over the specified number of rounds
        for i in range(1, rounds + 1):
            try:
                # Determine the current agent index
                agent_idx = i % 3

                # If we've completed a full round of all agents, increment the conversation round
                if (i > 2) and (agent_idx == 0):
                    convo_round += 1

                # Get the current agent
                agent = agents[agent_idx]

                # Append the agent's name and message to the output list
                out.append(f"Advisor: **{agent.name}**\nMessage:{agent.own_messages[convo_round]}\n------\n\n\n")
                df_data.append({
                    'Advisor': agent.name,
                    'Message': agent.own_messages[convo_round]
                })

            # If we've run out of messages for an agent, break the loop
            except IndexError:
                break

        # Join the output list into a single string and return it
        df = pd.DataFrame(df_data)
        return "".join(out), df

    def get_llm(self, model: str):
        """
        Get an instance of the OpenAI model.

        Parameters:
        model (str): The name of the OpenAI model.

        Returns:
        OpenAI: An instance of the OpenAI model.
        """
        return OpenAI(model=model, max_tokens= 3500)

    def get_program(self, llm, output_cls, prompt_template_str, verbose=False):
        """
        Get an instance of the OpenAIPydanticProgram.

        Parameters:
        llm (OpenAI): An instance of the OpenAI model.
        output_cls (BaseModel): The class of the output.
        prompt_template_str (str): The prompt template string.
        verbose (bool): Whether to print verbose output. Default is False.

        Returns:
        OpenAIPydanticProgram: An instance of the OpenAIPydanticProgram.
        """
        return OpenAIPydanticProgram.from_defaults(
            output_cls=output_cls,
            llm=llm,
            prompt_template_str=prompt_template_str,
            verbose=verbose,
        )

    def get_report(self, program, data):
        """
        Get a report from the program with error handling and retry logic.

        Parameters:
        program (OpenAIPydanticProgram): An instance of the OpenAIPydanticProgram.
        data (str): The data to analyze.

        Returns:
        Report: The generated report.
        """
        max_retries = 3
        retry_delay = 5  # seconds

        for attempt in range(max_retries):
            try:
                logging.info(f"Attempt {attempt + 1} to generate report")
                report = program(data=data)
                logging.info("Report generated successfully")
                return report
            except ValidationError as e:
                logging.error(f"Validation error occurred: {str(e)}")
                if attempt < max_retries - 1:
                    logging.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logging.error("Max retries reached. Unable to generate report.")
                    raise

        # This line should never be reached, but added for completeness
        raise Exception("Failed to generate report after multiple attempts")

    def split_and_group(self, text: str, max_sentences: int = 2) -> List[str]:
        """
        Splits the text into sentences and groups them.

        Args:
            text (str): The text to split and group.
            max_sentences (int, optional): The maximum number of sentences in a group. Defaults to 2.

        Returns:
            List[str]: The grouped sentences.
        """
        groups = []
        sentences = nltk.sent_tokenize(text)

        for i in range(len(sentences)):
            max_val = min(max_sentences + i, len(sentences)) if max_sentences != -1 else len(sentences)
            for j in range(i, max_val):
                grouped_text = ' '.join(sentences[i:j+1])
                groups.append(grouped_text)
        return groups

    def generate(self):
        op_data = []
        for key in self.report_dict.keys():
            op_data += [{
                'Opinion Type': key,
                "Opinion": x
            } for x in self.report_dict[key]]
        op_df = pd.DataFrame(op_data)

        sdf_list = []
        for i in range(len(self.cdf)):
            advisor = self.cdf.iloc[i]['Advisor']
            message = self.cdf.iloc[i]['Message']
            message_groups = self.split_and_group(message)
            sdf_list += [{"Advisor":advisor,
                        "grouped_message":x} for x in message_groups]

        sdf = pd.DataFrame(sdf_list)

        match_df = self.model.match(op_df["Opinion"], sdf["grouped_message"]).get_matches()

        out_df = op_df.merge(match_df, right_on = "From", left_on = "Opinion").merge(sdf, right_on = "grouped_message", left_on = "To").drop(labels = ["From", "To"], axis = 1).sort_values(by = "Opinion Type").reset_index(drop = True)
        out_df["Agent Source Message"] = out_df["grouped_message"]
        out_df.drop(labels = ["grouped_message"], axis = 1, inplace = True)
        return out_df

