import os
import json
import logging
from typing import List, Callable, Optional

from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType, load_tools
from langchain.memory import ConversationBufferMemory
import tiktoken
from utilities.utilities import handle_error
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    BaseMessage,
)

from utilities.sentiment import SentimentAnalyzer

sentiment_analyzer = SentimentAnalyzer()

OPENAI_MODEL = os.getenv("OPENAI_MODEL")
encoding = tiktoken.encoding_for_model(OPENAI_MODEL)


class AgentMessage:
    def __init__(self, content: str, sentiment_data: dict = None, metrics: dict = None) -> None:
        self.content = content
        self.sentiment_data = sentiment_data
        self.metrics = metrics

    def to_dict(self):
        message_dict = {
            "content": self.content,
            "sentiment_data": self.sentiment_data,
            "metrics": self.metrics
        }
        return message_dict


class DialogueAgent:
    def __init__(
        self,
        name: str,
        system_message: SystemMessage,
        model: ChatOpenAI(temperature = 1.5),
    ) -> None:
        self.name = name
        self.system_message = system_message
        self.model = model
        self.prefix = f"{self.name}: "
        self.reset()

        # empty container to put the history of each individual agent's messages
        self.own_messages = []
        self.messages = []

    def reset(self):
        self.message_history = ["Here is the conversation so far."]
        self.messages = []

    def send(self) -> str:
        """
        Applies the chatmodel to the message history
        and returns the message string
        """
        message = self.model(
            [
                self.system_message,
                HumanMessage(content="\n".join(self.message_history + [self.prefix])),
            ]
        )

        # add new message to the individual agent's history of messages
        message_content = message.content
        self.own_messages.append(message_content)

        self.messages.append(
            AgentMessage(
                content=message_content,
            )
        )
    
        return message.content

    def receive(self, name: str, message: str) -> None:
        """
        Concatenates {message} spoken by {name} into message history
        """
        self.message_history.append(f"{name}: {message}")
    
    # save individual agent's message to a txt file
    def save_own_messages(self, filename):
        
        # put the files in a separate directory to keep things organised
        directory = "output_files"

        # Create the directory if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(os.path.join(directory, filename), 'w') as f:
            for message in self.own_messages:
                f.write(f"{message}\n")


class DialogueSimulator:
    def __init__(
        self,
        agents: List[DialogueAgent],
        selection_function: Callable[[int, List[DialogueAgent]], int],
    ) -> None:
        self.agents = agents
        self._step = 0
        self.select_next_speaker = selection_function

        # container to put in the conversation history between agents
        self.conversation_history = []

    def reset(self):
        for agent in self.agents:
            agent.reset()

    def inject(self, name: str, message: str):
        """
        Initiates the conversation with a {message} from {name}
        """
        for agent in self.agents:
            agent.receive(name, message)

        # increment time
        self._step += 1
        
    def step(self) -> tuple[str, str]:
        try: 
            # 1. choose the next speaker
            speaker_idx = self.select_next_speaker(self._step, self.agents)
            speaker = self.agents[speaker_idx]

            # 2. next speaker sends message
            agent_message = speaker.send()
            message = agent_message.content
            # 3. everyone receives message
            for receiver in self.agents:
                receiver.receive(speaker.name, message)

            # 4. increment time
            self._step += 1

            # add the agent's name and message to the conversation history
            self.conversation_history.append(f"({speaker.name}): {message}")

            return speaker.name, agent_message, speaker_idx
    
        # error handling: raise exception if there's an issue with the step method
        except Exception as e:
            print("An error occurred in step method:", e)
            return None, None
    
    # save all messages in the conversation to a txt file (in a separate directory to keep things tidy)
    def save_conversation_history(self, filename):
        directory = "output_files"
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(os.path.join(directory, filename), 'w') as f:
            for message in self.conversation_history:
                f.write(f"{message}\n")


class DialogueAgentWithTools(DialogueAgent):
    """A dialogue agent equipped with additional tools."""

    def __init__(self, name: str, system_message: SystemMessage, model: ChatOpenAI(temperature = 1), tools, **tool_kwargs) -> None:
        """
        Initializes the DialogueAgentWithTools.

        Args:
            name: Name of the agent.
            system_message: Initial system message.
            model: Model used by the agent.
            tool_names: List of tools to be loaded for the agent.
            **tool_kwargs: Additional keyword arguments for tools.
        """
        tool_names = [x.name for x in tools]
        print(f"Initializing {name} with tools: {tool_names}")
        super().__init__(name, system_message, model)
        self.tools = tools
        self.total_tokens = 0

    def send(self) -> str:
        """
        Applies the chat model, tools, and returns the message string.
        """

        agent_chain = initialize_agent(
            self.tools, self.model, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=False,
            memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True),
            handle_parsing_errors=handle_error,
        )
        message_content = agent_chain.run("\n".join([self.system_message.content] + [str(x) for x in self.message_history] + [self.prefix]))
        message = AIMessage(content=message_content)

        token_count = len(encoding.encode(message.content))
        self.total_tokens += token_count

        print(f"Message token count: {token_count}, Total tokens: {self.total_tokens}")
        self.own_messages.append(message.content)
        agent_message = AgentMessage(
                content=message.content,
                sentiment_data = sentiment_analyzer.analyze_message(message.content),
            )
        self.messages.append(
            agent_message
        )
        #print("message content:", message.content)

        return agent_message