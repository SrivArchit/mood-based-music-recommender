# Load the model/tokenizer and predict emotions from text.
# This class will handle everything related to the fine-tuned text classification model.

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class EmotionDetector:
    
    def __init__(self, model_path, tokenizer_name="distilbert-base-uncased"):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model.eval() # Set model to evaluation mode
        print(f"✅ EmotionDetector initialized on device: {self.device}")

    def predict(self, text):
        
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, padding=True
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
        
        probabilities = torch.sigmoid(logits)
        # A threshold of 0.4 is used to determine if an emotion is present
        predictions = (probabilities > 0.4).int().cpu().numpy()
        predicted_indices = predictions[0].nonzero()[0]

        GOEMOTIONS_LABELS = [
            'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 
            'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 
            'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 
            'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 
            'relief', 'remorse', 'sadness', 'surprise', 'neutral'
        ]

        if len(predicted_indices) > 0:
            # Map the predicted indices back to their string labels
            return [GOEMOTIONS_LABELS[idx] for idx in predicted_indices]
        
        # Default to neutral if no strong emotion is detected
        return ["neutral"]