import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
import os
from tqdm import tqdm

class AudioDataset(Dataset):
    def __init__(self, data_dir, sample_rate=16000, transform=None):
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.transform = transform
        self.classes = sorted(os.listdir(data_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(data_dir, class_name)
            for file_name in os.listdir(class_dir):
                if file_name.endswith(('.wav', '.mp3', '.flac')):
                    self.samples.append((os.path.join(class_dir, file_name), class_name))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        audio_path, class_name = self.samples[idx]
        audio, _ = librosa.load(audio_path, sr=self.sample_rate)
        
        # Create mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=128,
            fmax=8000
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        if self.transform:
            mel_spec_db = self.transform(mel_spec_db)
        
        return torch.FloatTensor(mel_spec_db), self.class_to_idx[class_name]

class ModelTrainer:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        
    def train(self, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
            
            train_loss = train_loss / len(train_loader)
            train_acc = 100. * train_correct / train_total
            
            # Validation phase
            val_loss, val_acc = self.validate(val_loader, criterion)
            
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_model.pth')
    
    def validate(self, val_loader, criterion):
        self.model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        return val_loss, val_acc

def fine_tune_yamnet(data_dir, output_dir, num_epochs=10):
    # Load YAMNet model
    yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
    
    # Create a new model that uses YAMNet's embeddings
    class FineTunedYAMNet(tf.keras.Model):
        def __init__(self, num_classes):
            super(FineTunedYAMNet, self).__init__()
            self.yamnet = yamnet_model
            self.classifier = tf.keras.layers.Dense(num_classes, activation='softmax')
        
        def call(self, inputs):
            _, embeddings, _ = self.yamnet(inputs)
            return self.classifier(embeddings)
    
    # Create and compile model
    num_classes = len(os.listdir(data_dir))
    model = FineTunedYAMNet(num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Create data generators
    train_ds = tf.keras.utils.audio_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        output_sequence_length=16000,
        batch_size=32
    )
    
    val_ds = tf.keras.utils.audio_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        output_sequence_length=16000,
        batch_size=32
    )
    
    # Train model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=num_epochs,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(output_dir, 'yamnet_finetuned.h5'),
                save_best_only=True
            )
        ]
    )
    
    return model, history

if __name__ == "__main__":
    # Example usage
    data_dir = "path/to/your/audio/dataset"
    output_dir = "path/to/save/models"
    
    # Fine-tune CNN model
    from model import SoundClassifier
    cnn_model = SoundClassifier(num_classes=len(os.listdir(data_dir)))
    trainer = ModelTrainer(cnn_model)
    
    # Create datasets
    dataset = AudioDataset(data_dir)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Train CNN model
    trainer.train(train_loader, val_loader)
    
    # Fine-tune YAMNet
    yamnet_model, history = fine_tune_yamnet(data_dir, output_dir) 