# core/dialog.py
import os
import traceback
from typing import List, Callable
from utilities.utilities import handle_error, get_model
from utilities.sentiment import SentimentAnalyzer

sentiment_analyzer = SentimentAnalyzer(force_cpu=False)


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
        self.model = get_model(model_name, temperature)
        self.prefix = f"{self.name}: "
        self.current_round = 0
        self.reset()
        self.own_messages = []
        self.messages = []

    def reset(self):
        self.message_history = ["Here is the conversation so far."]
        self.messages = []
        self.current_round = 0

    def send(self) -> AgentMessage:
        """Generate response with full context"""
        
        # Use FULL history - no truncation
        conversation_history = "\n".join(self.message_history)
        
        try:
            # Extract system message content
            system_text = getattr(self.system_message, "content", str(self.system_message))
            
            # Minimal prompt structure - let the model behave naturally
            full_prompt = (
                f"{system_text}\n\n"
                f"{conversation_history}\n\n"
                f"{self.prefix}"
            )
            
            response = self.model.invoke(full_prompt)
            message_content = response if isinstance(response, str) else str(response)

        except Exception as e:
            print(f"[ERROR] {self.name}.send() failed: {e}")
            traceback.print_exc()
            message_content = "[ERROR: no response]"

        self.own_messages.append(message_content)

        # Create AgentMessage with sentiment analysis
        agent_message = AgentMessage(
            content=message_content,
            sentiment_data=sentiment_analyzer.analyze_message(message_content),
        )
        self.messages.append(agent_message)
        
        return agent_message

    def receive(self, name: str, message: str) -> None:
        """Receive message from another agent - FILTER OUT OWN MESSAGES"""
        
        # Don't add your own messages back to your history
        if name == self.name:
            return
        
        expected_entry = f"{name}: {message}"
        
        # Sanity check for duplicates
        if self.message_history and self.message_history[-1] == expected_entry:
            print(f"[WARNING] {self.name} received duplicate message from {name}")
            return
        
        self.message_history.append(expected_entry)

    def increment_round(self):
        """Called by simulator to track rounds"""
        self.current_round += 1

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
        self.agents_per_round = len(agents)

    def reset(self):
        for agent in self.agents:
            agent.reset()
        self._step = 0

    def inject(self, name: str, message: str):
        """Inject initial message (e.g., evaluation topic)"""
        facilitator_name = "Facilitator"
        for agent in self.agents:
            agent.receive(facilitator_name, message)

    def step(self) -> tuple[str, AgentMessage, int]:
        """Execute one conversation step with round tracking"""
        try:
            speaker_idx = self.select_next_speaker(self._step, self.agents)
            speaker = self.agents[speaker_idx]
            
            agent_message = speaker.send()
            
            # Broadcast to ALL agents
            for receiver in self.agents:
                receiver.receive(speaker.name, agent_message.content)
            
            # Store in conversation history
            self.conversation_history.append(f"{speaker.name}: {agent_message.content}")
            
            self._step += 1
            
            # Check if round just completed
            if self._step % self.agents_per_round == 0:
                for agent in self.agents:
                    agent.increment_round()
            
            return speaker.name, agent_message, speaker_idx
            
        except Exception as e:
            print(f"[ERROR] DialogueSimulator.step() failed: {e}")
            traceback.print_exc()
            fallback_msg = AgentMessage(content="[ERROR: step failed]")
            return "UNKNOWN", fallback_msg, 0

    def save_conversation_history(sxelf, filename):
        directory = "output_files"
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, filename), "w") as f:
            f.writelines(f"{msg}\n" for msg in self.conversation_history)


class DialogueAgentWithTools(DialogueAgent):
    def __init__(
            self, 
            name: str, 
            system_message, 
            model_name: str, 
            tools, 
            temperature: float = 0.3
    ) -> None:
        super().__init__(name, system_message, model_name, temperature)
        self.tools = tools
        self.total_tokens = 0

    def send(self) -> AgentMessage:
        """Same as base but with tools available"""
        
        conversation_history = "\n".join(self.message_history)
        
        try:
            system_text = getattr(self.system_message, "content", str(self.system_message))
            
            full_prompt = (
                f"{system_text}\n\n"
                f"{conversation_history}\n\n"
                f"{self.prefix}"
            )
            
            response = self.model.invoke(full_prompt)
            message_content = response if isinstance(response, str) else str(response)
            
        except Exception as e:
            print(f"[ERROR] {self.name}.send() with tools failed: {e}")
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
    def __init__(
            self, 
            name: str, 
            system_message, 
            model_name: str, 
            tools, 
            temperature: float = 0.3
    ) -> None:
        super().__init__(name, system_message, model_name, tools, temperature)
        self.own_sentiment = "neutral"

    def update_own_sentiment_awareness(self, own_sentiment: str):
        """Update agent's awareness of their own sentiment"""
        self.own_sentiment = own_sentiment

    def send(self) -> AgentMessage:
        """Send with own sentiment awareness"""
        
        conversation_history = "\n".join(self.message_history)
        
        try:
            system_text = getattr(self.system_message, "content", str(self.system_message))
            
            # Just add sentiment awareness naturally into the context
            sentiment_note = f"[Your recent sentiment has been {self.own_sentiment}]"
            
            full_prompt = (
                f"{system_text}\n\n"
                f"{sentiment_note}\n\n"
                f"{conversation_history}\n\n"
                f"{self.prefix}"
            )
            
            response = self.model.invoke(full_prompt)
            message_content = response if isinstance(response, str) else str(response)
            
        except Exception as e:
            print(f"[ERROR] {self.name}.send() with own sentiment failed: {e}")
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
    def __init__(
            self, 
            name: str, 
            system_message, 
            model_name: str, 
            tools, 
            temperature: float = 0.3
    ) -> None:
        super().__init__(name, system_message, model_name, tools, temperature)
        self.other_agents_sentiment = {}

    def update_others_sentiment_awareness(self, others_sentiment: dict):
        """Update agent's awareness of others' sentiment"""
        self.other_agents_sentiment = others_sentiment

    def send(self) -> AgentMessage:
        """Send with others' sentiment awareness"""
        
        conversation_history = "\n".join(self.message_history)
        
        try:
            system_text = getattr(self.system_message, "content", str(self.system_message))
            
            # Just add others' sentiment naturally into the context
            sentiment_note = ""
            if self.other_agents_sentiment:
                others_info = ", ".join([
                    f"{name}: {sentiment}" 
                    for name, sentiment in self.other_agents_sentiment.items()
                ])
                sentiment_note = f"[Other participants' recent sentiment: {others_info}]"
            
            full_prompt = (
                f"{system_text}\n\n"
                f"{sentiment_note}\n\n"
                f"{conversation_history}\n\n"
                f"{self.prefix}"
            )
            
            response = self.model.invoke(full_prompt)
            message_content = response if isinstance(response, str) else str(response)
            
        except Exception as e:
            print(f"[ERROR] {self.name}.send() with others sentiment failed: {e}")
            traceback.print_exc()
            message_content = "[ERROR: no response]"

        self.own_messages.append(message_content)
        
        agent_message = AgentMessage(
            content=message_content,
            sentiment_data=sentiment_analyzer.analyze_message(message_content),
        )
        self.messages.append(agent_message)
        
        return agent_message