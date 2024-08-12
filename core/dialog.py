import os
from typing import List, Callable

from langchain.schema import HumanMessage, SystemMessage
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
import tiktoken
from utilities.utilities import handle_error
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
        return {
            "content": self.content,
            "sentiment_data": self.sentiment_data,
            "metrics": self.metrics
        }


class DialogueAgent:
    def __init__(
        self,
        name: str,
        system_message: SystemMessage,
        model: ChatOpenAI,
    ) -> None:
        self.name = name
        self.system_message = system_message
        self.model = model
        self.prefix = f"{self.name}: "
        self.reset()
        self.own_messages = []
        self.messages = []

    def reset(self):
        self.message_history = ["Here is the conversation so far."]
        self.messages = []

    def send(self) -> str:
        message = self.model([
            self.system_message,
            HumanMessage(content="\n".join(self.message_history + [self.prefix])),
        ])
        message_content = message.content
        self.own_messages.append(message_content)
        self.messages.append(AgentMessage(content=message_content))
        return message_content

    def receive(self, name: str, message: str) -> None:
        self.message_history.append(f"{name}: {message}")

    def save_own_messages(self, filename):
        directory = "output_files"
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, filename), 'w') as f:
            f.writelines(f"{msg}\n" for msg in self.own_messages)


class DialogueSimulator:
    def __init__(
        self,
        agents: List[DialogueAgent],
        selection_function: Callable[[int, List[DialogueAgent]], int],
    ) -> None:
        self.agents = agents
        self._step = 0
        self.select_next_speaker = selection_function
        self.conversation_history = []

    def reset(self):
        for agent in self.agents:
            agent.reset()

    def inject(self, name: str, message: str):
        for agent in self.agents:
            agent.receive(name, message)
        self._step += 1

    def step(self) -> tuple[str, AgentMessage, int]:
        try:
            speaker_idx = self.select_next_speaker(self._step, self.agents)
            speaker = self.agents[speaker_idx]
            agent_message = speaker.send()
            for receiver in self.agents:
                receiver.receive(speaker.name, agent_message)
            self._step += 1
            self.conversation_history.append(f"({speaker.name}): {agent_message}")
            return speaker.name, agent_message, speaker_idx
        except Exception as e:
            print("An error occurred in step method:", e)
            return None, None, None

    def save_conversation_history(self, filename):
        directory = "output_files"
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, filename), 'w') as f:
            f.writelines(f"{msg}\n" for msg in self.conversation_history)


class DialogueAgentWithTools(DialogueAgent):
    def __init__(self, name: str, system_message: SystemMessage, model: ChatOpenAI, tools, **tool_kwargs) -> None:
        super().__init__(name, system_message, model)
        self.tools = tools
        self.total_tokens = 0

    def send(self) -> AgentMessage:
        agent_chain = initialize_agent(
            self.tools, self.model, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=False,
            memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True),
            handle_parsing_errors=handle_error,
        )
        message_content = agent_chain.run(
            "\n".join([self.system_message.content] + [str(x) for x in self.message_history] + [self.prefix])
        )
        token_count = len(encoding.encode(message_content))
        self.total_tokens += token_count
        print(f"Message token count: {token_count}, Total tokens: {self.total_tokens}")
        self.own_messages.append(message_content)
        agent_message = AgentMessage(
            content=message_content,
            sentiment_data=sentiment_analyzer.analyze_message(message_content),
        )
        self.messages.append(agent_message)
        return agent_message
