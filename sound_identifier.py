import sys
import numpy as np
import sounddevice as sd
import torch
import torchaudio
import librosa
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QLabel, QPushButton, QProgressBar, QComboBox)
from PyQt6.QtCore import QTimer, Qt
from ensemble_model import EnsembleSoundClassifier

class SoundIdentifier(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NoiseNinja - Sound Identifier")
        self.setGeometry(100, 100, 800, 600)
        
        # Initialize audio parameters
        self.sample_rate = 16000
        self.chunk_size = 1024
        self.stream = None
        self.is_recording = False
        
        # Initialize ensemble model
        self.model = EnsembleSoundClassifier(confidence_threshold=0.5)
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()
        
        # Create UI elements
        self.status_label = QLabel("Status: Ready")
        self.result_label = QLabel("Detected Sound: None")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setStyleSheet("font-size: 24px; margin: 20px;")
        
        # Add confidence bar
        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setValue(0)
        self.confidence_bar.setFormat("Confidence: %p%")
        
        # Add confidence threshold selector
        self.threshold_label = QLabel("Confidence Threshold:")
        self.threshold_selector = QComboBox()
        self.threshold_selector.addItems(['0.3', '0.4', '0.5', '0.6', '0.7', '0.8'])
        self.threshold_selector.setCurrentText('0.5')
        self.threshold_selector.currentTextChanged.connect(self.update_threshold)
        
        self.toggle_button = QPushButton("Start Recording")
        self.toggle_button.clicked.connect(self.toggle_recording)
        
        # Add widgets to layout
        layout.addWidget(self.status_label)
        layout.addWidget(self.result_label)
        layout.addWidget(self.confidence_bar)
        layout.addWidget(self.threshold_label)
        layout.addWidget(self.threshold_selector)
        layout.addWidget(self.toggle_button)
        
        main_widget.setLayout(layout)
        
        # Initialize audio stream
        self.setup_audio()
        
    def setup_audio(self):
        try:
            self.stream = sd.InputStream(
                channels=1,
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                callback=self.audio_callback
            )
        except Exception as e:
            self.status_label.setText(f"Error setting up audio: {str(e)}")
    
    def audio_callback(self, indata, frames, time, status):
        if status:
            print(f"Audio callback status: {status}")
        if self.is_recording:
            # Process audio data
            audio_data = indata[:, 0]
            self.process_audio(audio_data)
    
    def process_audio(self, audio_data):
        # Get prediction from ensemble model
        prediction, confidence = self.model.predict(audio_data, self.sample_rate)
        
        # Update UI
        self.result_label.setText(f"Detected Sound: {prediction}")
        self.confidence_bar.setValue(int(confidence * 100))
        
        # Update status based on confidence
        if confidence > float(self.threshold_selector.currentText()):
            self.status_label.setText("Status: High Confidence Detection")
        else:
            self.status_label.setText("Status: Low Confidence Detection")
    
    def update_threshold(self, value):
        self.model.confidence_threshold = float(value)
    
    def toggle_recording(self):
        if not self.is_recording:
            self.stream.start()
            self.is_recording = True
            self.toggle_button.setText("Stop Recording")
            self.status_label.setText("Status: Recording")
        else:
            self.stream.stop()
            self.is_recording = False
            self.toggle_button.setText("Start Recording")
            self.status_label.setText("Status: Stopped")
            self.result_label.setText("Detected Sound: None")
            self.confidence_bar.setValue(0)

def main():
    app = QApplication(sys.argv)
    window = SoundIdentifier()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 