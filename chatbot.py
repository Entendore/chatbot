import random
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import nltk

# Download necessary NLTK data
nltk.download('punkt')

class ChatBot:
    def __init__(self, name="ChatBot"):
        self.name = name
        
        # Sample data for training the chatbot
        self.training_data = [
            ("Hi", "greeting"),
            ("Hello", "greeting"),
            ("Hey there!", "greeting"),
            ("Bye", "farewell"),
            ("Goodbye", "farewell"),
            ("See you", "farewell"),
            ("How are you?", "how_are_you"),
            ("How are you doing?", "how_are_you"),
            ("What's your name?", "name"),
            ("Who are you?", "name"),
            ("Tell me a joke", "joke"),
            ("Tell me something funny", "joke"),
            ("What time is it?", "time"),
            ("What's the weather like?", "weather")
        ]
        
        # Train the intent classifier
        self.model = self.train_intent_classifier()

        # Responses for each intent
        self.responses = {
            "greeting": ["Hello there!", "Hi!", "Hey! How can I help you?", "Nice to meet you!"],
            "farewell": ["Goodbye!", "See you later!", "Take care!", "Bye!"],
            "how_are_you": ["I'm doing great, thanks for asking!", "All systems go!", "Feeling helpful today!"],
            "name": [f"I'm called {self.name}. What's your name?", f"My name is {self.name}, your friendly chatbot!"],
            "joke": ["Why don't scientists trust atoms? Because they make up everything!", 
                     "Why did the computer go to the doctor? It had a virus!"],
            "time": ["I can't check the time right now, but you can try asking your device.", 
                     "I’m sorry, I don’t have a clock at the moment!"],
            "weather": ["I don't know the current weather, but you can check it online!", 
                        "Unfortunately, I don't have a weather API right now."]
        }
    
    def train_intent_classifier(self):
        # Prepare training data
        X_train = [item[0] for item in self.training_data]
        y_train = [item[1] for item in self.training_data]
        
        # Create a machine learning pipeline
        model = make_pipeline(TfidfVectorizer(), SVC(kernel='linear'))
        
        # Train the classifier
        model.fit(X_train, y_train)
        
        return model
    
    def classify_intent(self, message):
        # Use the trained model to classify intent
        return self.model.predict([message])[0]
    
    def extract_entities(self, message):
        # Simple entity extraction using regex or basic patterns
        entities = {}
        
        # Check for specific entities like time, date, etc.
        time_match = re.search(r"\b(?:what time is it|time|clock)\b", message.lower())
        if time_match:
            entities["time"] = "time"
        
        # Example: Extracting numbers as potential entities
        number_match = re.findall(r'\d+', message)
        if number_match:
            entities["numbers"] = number_match
        
        # You can add more complex patterns and rules here
        
        return entities
    
    def get_response(self, message):
        # Classify the intent
        intent = self.classify_intent(message)
        
        # Extract entities (currently just simple ones like time, numbers)
        entities = self.extract_entities(message)
        
        # Customize the response based on entities if needed
        if "time" in entities:
            return "Sorry, I can't tell the time right now. Please check a clock or your device."

        # Get the appropriate response based on the intent
        response = random.choice(self.responses.get(intent, self.responses["farewell"]))
        
        return response
    
    def chat(self):
        print(f"{self.name}: Hi! Type 'quit' to end the chat.")
        while True:
            user_input = input("You: ")
            if user_input.strip().lower() == "quit":
                print(f"{self.name}: {random.choice(self.responses['farewell'])}")
                break
            response = self.get_response(user_input)
            print(f"{self.name}: {response}")


if __name__ == "__main__":
    bot = ChatBot("RoboHelper")
    bot.chat()
