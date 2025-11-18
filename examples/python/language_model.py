"""
Language Model Implementation
Supports both statistical and neural network models
"""
import pickle
import numpy as np
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


class LanguageModel:
    """Base class for language models supporting multiple model types"""

    def __init__(self, model_type, model_path):
        """
        Initialize a language model

        Args:
            model_type: Type of model ('statistical' or 'neural_network')
            model_path: Path to the saved model file
        """
        self.model_type = model_type
        self.model = self.load_model(model_path)
        self.tokenizer = Tokenizer()  # Tokenizer for neural network model

    def load_model(self, model_path):
        """
        Load a model from disk

        Args:
            model_path: Path to the model file

        Returns:
            Loaded model object
        """
        if self.model_type == 'statistical':
            with open(model_path, 'rb') as file:
                return pickle.load(file)
        elif self.model_type == 'neural_network':
            return load_model(model_path)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def generate_response(self, input_text):
        """
        Generate a response for the given input text

        Args:
            input_text: Input text to generate response for

        Returns:
            Generated response
        """
        if self.model_type == 'statistical':
            # Assuming the statistical model has a predict method
            return self.model.predict(input_text)
        elif self.model_type == 'neural_network':
            # Tokenize the input text
            sequences = self.tokenizer.texts_to_sequences([input_text])
            data = pad_sequences(sequences, maxlen=100)  # Assuming a maxlen of 100

            # Predict the next token
            predicted = self.model.predict(data)
            predicted_token_id = np.argmax(predicted[0])

            # Convert the token ID back to text
            predicted_token = self.tokenizer.index_word.get(predicted_token_id, '<UNK>')
            return predicted_token
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")


def rnn_encoder_decoder(src_seq, tgt_seq):
    """
    RNN Encoder-Decoder for Statistical Machine Translation.
    Takes as input a source sequence and a target sequence and returns the
    predicted target sequence.

    Arguments:
        src_seq (list): Source sequence to encode
        tgt_seq (list): Target sequence to decode

    Returns:
        tgt_seq_pred (list): Predicted target sequence
    """
    from . import rnn_encoder, rnn_decoder

    encoder = rnn_encoder.RNNEncoder()  # Instantiate RNN encoder
    decoder = rnn_decoder.RNNDecoder()   # Instantiate RNN decoder

    encoded_src_seq = encoder.encode(src_seq)  # Encode source sequence
    target_indices, attn_weights, _ = decoder.decode(tgt_seq, encoded_src_seq)  # Decode target sequence

    # Reconstruct target sequence from predicted indices and context vectors
    tgt_seq_pred = []
    for i, (idx, cnx) in enumerate(zip(target_indices, attn_weights.T)):
        cnx = np.squeeze(cnx)  # Ensure cnx is 1D
        tgt_seq_pred.append(vocab.index2token(np.argmax(cnx)))

    return tgt_seq_pred
