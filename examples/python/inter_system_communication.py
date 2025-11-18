"""
Inter-System Communication Language
Facilitates communication between different language models
"""
import numpy as np
from .language_model import LanguageModel


class InterSystemCommunicationLanguage:
    """Manages communication between multiple language models"""

    def __init__(self):
        """Initialize the inter-system communication system"""
        self.language_models = []

    def add_language_model(self, model_type, model_path):
        """
        Add a new language model to the system

        Args:
            model_type: Type of model ('statistical' or 'neural_network')
            model_path: Path to the saved model file
        """
        self.language_models.append(LanguageModel(model_type, model_path))

    def communicate(self, input_text):
        """
        Communicate with all language models and generate responses

        Args:
            input_text: Input text to process

        Returns:
            List of responses from all models
        """
        responses = []
        for model in self.language_models:
            response = model.generate_response(input_text)
            responses.append(response)
        return responses

    def optimize_communication(self, responses):
        """
        Optimize communication by combining responses from different models
        Uses ensemble averaging for optimization

        Args:
            responses: List of responses from different models

        Returns:
            Optimized response
        """
        # For simplicity, use ensemble averaging
        # This assumes all responses are of the same length and can be averaged element-wise
        try:
            return np.mean(responses, axis=0)
        except:
            # If responses can't be averaged (e.g., strings), return the first response
            return responses[0] if responses else None


# Example usage
if __name__ == '__main__':
    system = InterSystemCommunicationLanguage()
    system.add_language_model('statistical', 'models/statistical_model.pkl')
    system.add_language_model('neural_network', 'models/neural_network_model.h5')

    input_text = "Hello, how are you?"
    raw_responses = system.communicate(input_text)
    optimized_response = system.optimize_communication(raw_responses)

    print(f"Optimized Response: {optimized_response}")
