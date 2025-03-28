'''
This is the .py file which contains:
1. The ConversationManager class which is responsible for managing the conversation history and chat completions.
- The class has methods to enforce token budget, count tokens, set persona, set custom system message, update system message in history, load conversation history and save conversation history. The class also has a method to generate chat completions.
2. 
'''

from openai import OpenAI
import tiktoken
import json
from datetime import datetime
import os
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

DEFAULT_API_KEY = os.getenv("TOGETHER_API_KEY")
DEFAULT_BASE_URL = "https://api.together.xyz/v1"
DEFAULT_MODEL = "meta-llama/Llama-3-8b-chat-hf"
DEFAULT_TEMPERATURE=0.5
DEFAULT_MAX_TOKENS=128
DEFAULT_TOKEN_BUDGET=1280
DEFAULT_HISTORY_FILE = "conversation_history.json"

class ConversationManager:
    def __init__(self, api_key=None, base_url=None, model=None, temperature=None, max_tokens=None, token_budget=None, history_file=None
                ):
        if not api_key:
            api_key = DEFAULT_API_KEY
        if not base_url:
            base_url = DEFAULT_BASE_URL
        
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model if model else DEFAULT_MODEL
        self.temperature = temperature if temperature else DEFAULT_TEMPERATURE
        self.max_tokens = max_tokens if max_tokens else DEFAULT_MAX_TOKENS
        self.token_budget = token_budget if token_budget else DEFAULT_TOKEN_BUDGET
        self.history_file = history_file if history_file else DEFAULT_HISTORY_FILE
        self.system_messages = {
            "sassy": "You are a sassy assistant who is fed up with answering questions.",
            "concise": "You are a straightforward and concise assistant who is always ready to help.",
            "comedian": "You are a a stand-up comedian who specializes in wine jokes.",
            "custom": "Enter your custom system message here."
        }
        # Default system message
        self.system_message = self.system_messages["concise"]
        
        # Load conversation history
        self.conversation_history = []

        self.load_conversation_history()

    def chat_completion(self, prompt, temperature=None, max_tokens=None):
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens

        # Add user message first
        self.conversation_history.append({"role": "user", "content": prompt})

        # Enforce token budget *after* adding the user message
        self.enforce_token_budget()

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.conversation_history,
                temperature=temperature,
                max_tokens=max_tokens
            )
            ai_response = response.choices[0].message.content
            self.conversation_history.append({"role": "assistant", "content": ai_response})
            self.save_conversation_history()
            return ai_response
        except Exception as e:
            error_message = f"Error generating completion: {str(e)}"
            print(error_message)
            return None

    def enforce_token_budget(self):
        """ Ensures that the total token count does not exceed the token budget. """
        try:
            while self.total_tokens_used() > self.token_budget:
                if len(self.conversation_history) <= 1:
                    break  # Never remove the system message
                # Remove the *oldest* non-system message
                self.conversation_history.pop(1)
        except Exception as e:
            print(f"Error enforcing token budget: {str(e)}")

    def count_tokens(self, text):
        """ Counts tokens for a given text. """
        try:
            encoding = tiktoken.encoding_for_model(self.model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))

    def total_tokens_used(self):
        """ Computes total tokens used, considering OpenAI's message format. """
        total_tokens = 0
        for message in self.conversation_history:
            total_tokens += self.count_tokens(message['content'])
            total_tokens += 4  # Extra tokens for metadata per message (approx.)
        return total_tokens
    
    def set_persona(self, persona):
        if persona in self.system_messages:
            self.system_message = self.system_messages[persona]
            self.update_system_message_in_history()
        else:
            raise ValueError(f"Unknown persona: {persona}. Available personas are: {list(self.system_messages.keys())}")

    def set_custom_system_message(self, custom_message):
        if not custom_message:
            raise ValueError("Custom message cannot be empty.")
        self.system_messages['custom'] = custom_message
        self.set_persona('custom')

    def update_system_message_in_history(self):
        if self.conversation_history and self.conversation_history[0]["role"] == "system":
            self.conversation_history[0]["content"] = self.system_message
        else:
            self.conversation_history.insert(0, {"role": "system", "content": self.system_message})

    def load_conversation_history(self):
        try:
            with open(self.history_file, "r") as file:
                self.conversation_history = json.load(file)
        except FileNotFoundError:
            # Start with an initial history containing a single system message
            self.conversation_history = [{"role": "system", "content": self.system_message}]
        except json.JSONDecodeError:
            print("Error reading the conversation history file. Starting with an initial history.")
            self.conversation_history = [{"role": "system", "content": self.system_message}]
            
    def save_conversation_history(self):
        try:
            with open(self.history_file, "w") as file:
                json.dump(self.conversation_history, file, indent=4)
        except IOError as i:
            print(f"A file operation error occurred while saving the conversation history: {i}")
        except Exception as e:
            print(f"A general error occurred while saving the conversation history: {e}")

    def reset_conversation_history(self):
        self.conversation_history = [{"role": "system", "content": self.system_message}]
        try:
            self.save_conversation_history()
        except Exception as e:
            print(f"A general error occurred while resetting the conversation history: {e}")    

