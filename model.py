import torch
import torch.nn as nn
import torch.nn.functional as F

class SoundClassifier(nn.Module):
    def __init__(self, num_classes=527):  # AudioSet has 527 sound classes
        super(SoundClassifier, self).__init__()
        
        # CNN layers for processing mel spectrograms
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling layers
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Input shape: (batch_size, 1, 128, 128) - mel spectrogram
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten the output
        x = x.view(-1, 128 * 16 * 16)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

def load_model(model_path=None):
    model = SoundClassifier()
    if model_path and torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path))
        model = model.cuda()
    elif model_path:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict_sound(model, mel_spectrogram):
    """
    Predict the sound class from a mel spectrogram
    Args:
        model: The trained SoundClassifier model
        mel_spectrogram: numpy array of shape (128, 128)
    Returns:
        predicted_class: int, the predicted class index
        confidence: float, the confidence score
    """
    # Convert to tensor and add batch and channel dimensions
    x = torch.from_numpy(mel_spectrogram).float()
    x = x.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    
    if torch.cuda.is_available():
        x = x.cuda()
    
    with torch.no_grad():
        outputs = model(x)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    return predicted.item(), confidence.item() 