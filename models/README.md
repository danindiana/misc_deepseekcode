# Models Directory

This directory contains trained language models for the Inter-System Communication Language framework.

## Model Files

### Statistical Models
- `statistical_model.pkl` - Serialized statistical language model

### Neural Network Models
- `neural_network_model.h5` - Keras/TensorFlow neural network model
- `neural_network_model.pt` - PyTorch model (optional)

## Model Formats

### Python (Pickle)
```python
import pickle

# Save model
with open('statistical_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load model
with open('statistical_model.pkl', 'rb') as f:
    model = pickle.load(f)
```

### Keras/TensorFlow (H5)
```python
from keras.models import load_model, save_model

# Save model
model.save('neural_network_model.h5')

# Load model
model = load_model('neural_network_model.h5')
```

### PyTorch (PT)
```python
import torch

# Save model
torch.save(model.state_dict(), 'model.pt')

# Load model
model.load_state_dict(torch.load('model.pt'))
```

## Creating Your Own Models

See the training examples in `examples/*/training/` for how to train and save your own models.

## Note

Model files are excluded from git (see `.gitignore`). You'll need to train or download models separately.
