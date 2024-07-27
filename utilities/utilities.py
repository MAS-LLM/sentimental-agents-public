# Standard Library Imports
from typing import List, Dict, Any
import os

# Third-party Library Imports
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
#import spacy
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate



from langchain_community.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage


OPENAI_MODEL = os.getenv("OPENAI_MODEL")

# Initialize NLP Model
#nlp = spacy.load("en_core_web_sm")


def handle_error(error: Exception) -> str:
    """Handle errors and return a truncated message.

    Parameters:
        error (Exception): The Exception object.

    Returns:
        str: Truncated error message.
    """
    return str(error)[:50]


def get_sentiment(text: str) -> float:
    """Calculate and return the sentiment score of the given text.

    Parameters:
        text (str): Input text for sentiment analysis.

    Returns:
        float: Sentiment score.
    """
    analyzer = SentimentIntensityAnalyzer()
    vs = analyzer.polarity_scores(str(text))
    return vs['compound']


def generate_content_from_template(name: str, template: str, word_limit: int = None, extra_vars: Dict[str, Any] = None) -> str:
    """Generate content using a specified template.

    Parameters:
        name (str): Name of the agent.
        template (str): The template to be filled.
        word_limit (int): Limit for word count.
        extra_vars (Dict[str, Any]): Extra variables to be used in formatting.

    Returns:
        str: Generated content.
    """
    prompt_vars = {'name': name, 'word_limit': word_limit}
    if extra_vars:
        prompt_vars.update(extra_vars)

    prompt = [
        HumanMessage(
            content=template.format(**prompt_vars)
        ),
    ]
    return ChatOpenAI(model_name=OPENAI_MODEL, temperature=1.0)(prompt).content


#def extract_names(text: str) -> List[str]:
#    """Extract names from the given text.
#
#    Parameters:
#        text (str): Input text from which to extract names.
#
#    Returns:
#        List[str]: List of names.
#    """
#    doc = nlp(text)
#    return [entity.text for entity in doc.ents if entity.label_ == "PERSON"]


def summarise_document(messages_history: Any, temperature = 0) -> str:
    """Generate a summary for the provided messages.

    Parameters:
        messages_history (Any): History of messages to be summarized.

    Returns:
        str: Summary of the document.
    """
    try:
        summary_template = """Write a concise summary of the following messages:

        {messages_history}

        Answer in bullet points. 
        Don't use corporate jargon.

        """
        llm = ChatOpenAI(model = OPENAI_MODEL,temperature=temperature, max_tokens=256)
        prompt = PromptTemplate(template=summary_template, input_variables=["messages_history"])
        chain = LLMChain(llm=llm, prompt=prompt)

        input_data = {
            "messages_history": "\n".join([str(x) for x in messages_history]) if isinstance(messages_history, list) else messages_history
        }
        return chain.run(input_data)

    except Exception as e:
        print(f"An error occurred while summarizing the document: {handle_error(e)}")
        return None
