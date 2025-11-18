"""
Unit tests for the LanguageModel class
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock


class TestLanguageModel:
    """Test suite for LanguageModel class"""

    @pytest.fixture
    def mock_statistical_model(self, tmp_path):
        """Create a mock statistical model"""
        import pickle
        model_path = tmp_path / "statistical_model.pkl"
        mock_model = Mock()
        mock_model.predict = Mock(return_value="statistical response")
        with open(model_path, 'wb') as f:
            pickle.dump(mock_model, f)
        return str(model_path)

    @pytest.fixture
    def mock_neural_model(self, tmp_path):
        """Create a mock neural network model"""
        model_path = tmp_path / "neural_model.h5"
        # Create a minimal mock model file
        model_path.touch()
        return str(model_path)

    def test_language_model_initialization_statistical(self, mock_statistical_model):
        """Test LanguageModel initialization with statistical model"""
        from examples.python.language_model import LanguageModel

        model = LanguageModel('statistical', mock_statistical_model)
        assert model.model_type == 'statistical'
        assert model.model is not None

    @patch('examples.python.language_model.load_model')
    def test_language_model_initialization_neural(self, mock_load, mock_neural_model):
        """Test LanguageModel initialization with neural network model"""
        from examples.python.language_model import LanguageModel

        mock_load.return_value = Mock()
        model = LanguageModel('neural_network', mock_neural_model)
        assert model.model_type == 'neural_network'

    def test_language_model_unsupported_type(self):
        """Test LanguageModel with unsupported model type"""
        from examples.python.language_model import LanguageModel

        with pytest.raises(ValueError, match="Unsupported model type"):
            LanguageModel('unsupported_type', 'fake_path.pkl')

    def test_generate_response_statistical(self, mock_statistical_model):
        """Test response generation with statistical model"""
        from examples.python.language_model import LanguageModel

        model = LanguageModel('statistical', mock_statistical_model)
        response = model.generate_response("Hello")
        assert response == "statistical response"

    @patch('examples.python.language_model.load_model')
    @patch('examples.python.language_model.Tokenizer')
    def test_generate_response_neural(self, mock_tokenizer, mock_load, mock_neural_model):
        """Test response generation with neural network model"""
        from examples.python.language_model import LanguageModel

        # Setup mocks
        mock_model = Mock()
        mock_model.predict = Mock(return_value=np.array([[0.1, 0.9, 0.0]]))
        mock_load.return_value = mock_model

        mock_tok = Mock()
        mock_tok.texts_to_sequences = Mock(return_value=[[1, 2, 3]])
        mock_tok.index_word = {0: 'hello', 1: 'world', 2: '<UNK>'}
        mock_tokenizer.return_value = mock_tok

        model = LanguageModel('neural_network', mock_neural_model)
        response = model.generate_response("Hello")

        # Should return the token with highest probability (index 1)
        assert response == 'world'


class TestInterSystemCommunication:
    """Test suite for InterSystemCommunicationLanguage class"""

    def test_initialization(self):
        """Test InterSystemCommunicationLanguage initialization"""
        from examples.python.inter_system_communication import InterSystemCommunicationLanguage

        system = InterSystemCommunicationLanguage()
        assert system.language_models == []

    @patch('examples.python.inter_system_communication.LanguageModel')
    def test_add_language_model(self, mock_lm):
        """Test adding language models"""
        from examples.python.inter_system_communication import InterSystemCommunicationLanguage

        system = InterSystemCommunicationLanguage()
        system.add_language_model('statistical', 'fake_path.pkl')

        assert len(system.language_models) == 1
        mock_lm.assert_called_once_with('statistical', 'fake_path.pkl')

    @patch('examples.python.inter_system_communication.LanguageModel')
    def test_communicate(self, mock_lm):
        """Test communication with multiple models"""
        from examples.python.inter_system_communication import InterSystemCommunicationLanguage

        # Setup mocks
        mock_model1 = Mock()
        mock_model1.generate_response = Mock(return_value="response1")
        mock_model2 = Mock()
        mock_model2.generate_response = Mock(return_value="response2")

        mock_lm.side_effect = [mock_model1, mock_model2]

        system = InterSystemCommunicationLanguage()
        system.add_language_model('statistical', 'path1.pkl')
        system.add_language_model('neural_network', 'path2.h5')

        responses = system.communicate("Hello")

        assert len(responses) == 2
        assert responses == ["response1", "response2"]

    def test_optimize_communication_numeric(self):
        """Test communication optimization with numeric responses"""
        from examples.python.inter_system_communication import InterSystemCommunicationLanguage

        system = InterSystemCommunicationLanguage()
        responses = [np.array([1.0, 2.0, 3.0]), np.array([2.0, 3.0, 4.0])]

        optimized = system.optimize_communication(responses)

        np.testing.assert_array_almost_equal(optimized, np.array([1.5, 2.5, 3.5]))

    def test_optimize_communication_strings(self):
        """Test communication optimization with string responses"""
        from examples.python.inter_system_communication import InterSystemCommunicationLanguage

        system = InterSystemCommunicationLanguage()
        responses = ["response1", "response2"]

        optimized = system.optimize_communication(responses)

        # Should return first response when can't average
        assert optimized == "response1"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
