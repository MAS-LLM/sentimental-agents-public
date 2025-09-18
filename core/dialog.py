import os
import traceback
from typing import List, Callable
from utilities.utilities import handle_error, get_model
from utilities.sentiment import SentimentAnalyzer

sentiment_analyzer = SentimentAnalyzer()


class AgentMessage:
    def __init__(self, content: str, sentiment_data: dict = None, metrics: dict = None) -> None:
        self.content = content
        self.sentiment_data = sentiment_data
        self.metrics = metrics

    def to_dict(self):
        return {
            "content": self.content,
            "sentiment_data": self.sentiment_data,
            "metrics": self.metrics,
        }


class DialogueAgent:
    def __init__(
            self,
            name: str,
            system_message,
            model_name: str = "llama3",
            temperature: float = 0.3,
    ) -> None:
        self.name = name
        self.system_message = system_message
        self.model = get_model(model_name, temperature)  # OllamaLLM instance
        self.prefix = f"{self.name}: "
        self.reset()
        self.own_messages = []
        self.messages = []

    def reset(self):
        self.message_history = ["Here is the conversation so far."]
        self.messages = []

    def send(self) -> AgentMessage:  # Changed return type to match others
        short_context = self.message_history[-2:] if len(self.message_history) > 2 else self.message_history
        limit_instruction = "\nRespond in no more than 2 sentences."
        prompt = "\n".join(short_context + [self.prefix]) + limit_instruction

        try:
            # Robust system message handling
            system_text = getattr(self.system_message, "content", str(self.system_message))
            response = self.model.invoke(system_text + "\n" + prompt)
            message_content = response if isinstance(response, str) else str(response)

            # Debug print to see what's being generated
            print(f"DEBUG: {self.name} generated: '{message_content[:50]}...'")

        except Exception as e:
            print(f"Error in {self.name}.send():", e)
            traceback.print_exc()
            message_content = "[ERROR: no response]"

        self.own_messages.append(message_content)

        # Create AgentMessage with sentiment analysis
        agent_message = AgentMessage(
            content=message_content,
            sentiment_data=sentiment_analyzer.analyze_message(message_content),
        )
        self.messages.append(agent_message)
        return agent_message  # Return AgentMessage object

    def receive(self, name: str, message: str) -> None:
        self.message_history.append(f"{name}: {message}")

    def save_own_messages(self, filename):
        directory = "output_files"
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, filename), "w") as f:
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
            print("Error in step():", e)
            traceback.print_exc()
            fallback_msg = AgentMessage(content="[ERROR: step failed]")
            return "UNKNOWN", fallback_msg, 0

    def save_conversation_history(self, filename):
        directory = "output_files"
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, filename), "w") as f:
            f.writelines(f"{msg}\n" for msg in self.conversation_history)


class DialogueAgentWithTools(DialogueAgent):
    def __init__(self, name: str, system_message, model_name: str, tools, temperature: float = 0.3) -> None:
        super().__init__(name, system_message, model_name, temperature)
        self.tools = tools
        self.total_tokens = 0

    def send(self) -> AgentMessage:
        short_context = self.message_history[-2:] if len(self.message_history) > 2 else self.message_history
        limit_instruction = "\nRespond in no more than 2 sentences."

        system_text = getattr(self.system_message, "content", str(self.system_message))
        prompt = "\n".join([system_text] + short_context + [self.prefix]) + limit_instruction

        try:
            response = self.model.invoke(prompt)
            message_content = response if isinstance(response, str) else str(response)
        except Exception as e:
            print(f"Error in {self.name}.send() with tools:", e)
            traceback.print_exc()
            message_content = "[ERROR: no response]"

        self.own_messages.append(message_content)
        agent_message = AgentMessage(
            content=message_content,
            sentiment_data=sentiment_analyzer.analyze_message(message_content),
        )
        self.messages.append(agent_message)
        return agent_message


class DialogueAgentWithOwnSentimentFeedback(DialogueAgentWithTools):
    def __init__(self, name: str, system_message, model_name: str, tools, temperature: float = 0.3) -> None:
        super().__init__(name, system_message, model_name, tools, temperature)
        self.own_sentiment = "neutral"

    def update_own_sentiment_awareness(self, own_sentiment: str):
        """Update agent's awareness of their own sentiment only"""
        self.own_sentiment = own_sentiment

    def send(self) -> AgentMessage:
        short_context = self.message_history[-2:] if len(self.message_history) > 2 else self.message_history
        limit_instruction = "\nRespond in no more than 2 sentences."

        system_text = getattr(self.system_message, "content", str(self.system_message))

        # Add own sentiment feedback to prompt
        sentiment_context = f"\n\nSentiment Awareness: Your recent tone has been {self.own_sentiment}. Consider this in your response."

        prompt = "\n".join([system_text] + short_context + [sentiment_context, self.prefix]) + limit_instruction

        try:
            response = self.model.invoke(prompt)
            message_content = response if isinstance(response, str) else str(response)
        except Exception as e:
            print(f"Error in {self.name}.send() with own sentiment feedback:", e)
            traceback.print_exc()
            message_content = "[ERROR: no response]"

        self.own_messages.append(message_content)
        agent_message = AgentMessage(
            content=message_content,
            sentiment_data=sentiment_analyzer.analyze_message(message_content),
        )
        self.messages.append(agent_message)
        return agent_message


class DialogueAgentWithOthersSentimentFeedback(DialogueAgentWithTools):
    def __init__(self, name: str, system_message, model_name: str, tools, temperature: float = 0.3) -> None:
        super().__init__(name, system_message, model_name, tools, temperature)
        self.other_agents_sentiment = {}

    def update_others_sentiment_awareness(self, others_sentiment: dict):
        """Update agent's awareness of others' sentiment only"""
        self.other_agents_sentiment = others_sentiment

    def send(self) -> AgentMessage:
        short_context = self.message_history[-2:] if len(self.message_history) > 2 else self.message_history
        limit_instruction = "\nRespond in no more than 2 sentences."

        system_text = getattr(self.system_message, "content", str(self.system_message))

        # Add others' sentiment feedback to prompt
        sentiment_context = ""
        if self.other_agents_sentiment:
            others_info = ", ".join([f"{name}: {sentiment}" for name, sentiment in self.other_agents_sentiment.items()])
            sentiment_context = f"\n\nSentiment Awareness: Other participants' recent tones: {others_info}. Consider this emotional context in your response."

        prompt = "\n".join([system_text] + short_context + [sentiment_context, self.prefix]) + limit_instruction

        try:
            response = self.model.invoke(prompt)
            message_content = response if isinstance(response, str) else str(response)
        except Exception as e:
            print(f"Error in {self.name}.send() with others sentiment feedback:", e)
            traceback.print_exc()
            message_content = "[ERROR: no response]"

        self.own_messages.append(message_content)
        agent_message = AgentMessage(
            content=message_content,
            sentiment_data=sentiment_analyzer.analyze_message(message_content),
        )
        self.messages.append(agent_message)
        return agent_message