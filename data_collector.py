import sounddevice as sd
import numpy as np
import librosa
import os
import yaml
import time
import json
from datetime import datetime
import geopy
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QLabel, QPushButton, QComboBox, QProgressBar)
from PyQt6.QtCore import QTimer, Qt

class LocationManager:
    def __init__(self, config):
        self.config = config
        self.geolocator = Nominatim(user_agent="NoiseNinja")
        self.last_location = self._load_last_location()
    
    def _load_last_location(self):
        if self.config['gps']['cache_location']:
            try:
                with open(self.config['gps']['last_location_file'], 'r') as f:
                    return json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                return None
        return None
    
    def _save_last_location(self, location_data):
        if self.config['gps']['cache_location']:
            with open(self.config['gps']['last_location_file'], 'w') as f:
                json.dump(location_data, f)
    
    def get_current_location(self):
        try:
            # Get location from system
            location = self.geolocator.geocode("me", timeout=self.config['gps']['timeout'])
            
            if location:
                location_data = {
                    'latitude': location.latitude,
                    'longitude': location.longitude,
                    'altitude': location.altitude if self.config['gps']['include_altitude'] else None,
                    'address': location.address if self.config['gps']['include_address'] else None,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Save location for caching
                self._save_last_location(location_data)
                return location_data
            
            # If location service fails, return cached location
            if self.last_location:
                return self.last_location
            
            return None
            
        except GeocoderTimedOut:
            if self.last_location:
                return self.last_location
            return None

class DataCollector(QMainWindow):
    def __init__(self, config_path='config.yaml'):
        super().__init__()
        self.setWindowTitle("NoiseNinja - Data Collector")
        self.setGeometry(100, 100, 800, 600)
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize audio parameters
        self.sample_rate = self.config['data_collection']['sample_rate']
        self.chunk_size = self.config['data_collection']['chunk_size']
        self.recording_duration = self.config['data_collection']['recording_duration']
        self.silence_threshold = self.config['data_collection']['silence_threshold']
        self.min_silence_duration = self.config['data_collection']['min_silence_duration']
        
        # Initialize location manager
        self.location_manager = LocationManager(self.config['data_collection'])
        
        # Initialize recording state
        self.is_recording = False
        self.audio_buffer = []
        self.silence_start = None
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()
        
        # Create UI elements
        self.status_label = QLabel("Status: Ready")
        self.level_label = QLabel("Audio Level: 0 dB")
        self.location_label = QLabel("Location: Not available")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        
        # Create class selector
        self.class_label = QLabel("Select Sound Class:")
        self.class_selector = QComboBox()
        self.class_selector.addItems(self._get_existing_classes())
        self.class_selector.setEditable(True)
        
        # Create record button
        self.record_button = QPushButton("Start Recording")
        self.record_button.clicked.connect(self.toggle_recording)
        
        # Add widgets to layout
        layout.addWidget(self.status_label)
        layout.addWidget(self.level_label)
        layout.addWidget(self.location_label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.class_label)
        layout.addWidget(self.class_selector)
        layout.addWidget(self.record_button)
        
        main_widget.setLayout(layout)
        
        # Initialize audio stream
        self.setup_audio()
        
        # Create output directory if it doesn't exist
        os.makedirs(self.config['data_collection']['output_directory'], exist_ok=True)
        
        # Update location periodically
        self.location_timer = QTimer()
        self.location_timer.timeout.connect(self.update_location)
        self.location_timer.start(5000)  # Update every 5 seconds
        self.update_location()
    
    def update_location(self):
        if self.config['data_collection']['gps']['enabled']:
            location = self.location_manager.get_current_location()
            if location:
                lat = location['latitude']
                lon = location['longitude']
                if self.config['data_collection']['gps']['location_format'] == 'degrees_minutes_seconds':
                    lat_str = self._decimal_to_dms(lat, 'lat')
                    lon_str = self._decimal_to_dms(lon, 'lon')
                else:
                    lat_str = f"{lat:.6f}"
                    lon_str = f"{lon:.6f}"
                
                location_text = f"Location: {lat_str}, {lon_str}"
                if location.get('address'):
                    location_text += f"\nAddress: {location['address']}"
                self.location_label.setText(location_text)
            else:
                self.location_label.setText("Location: Not available")
    
    def _decimal_to_dms(self, decimal, lat_or_lon):
        """Convert decimal degrees to degrees, minutes, seconds format"""
        degrees = int(decimal)
        minutes = int((decimal - degrees) * 60)
        seconds = (decimal - degrees - minutes/60) * 3600
        
        direction = 'N' if lat_or_lon == 'lat' and decimal >= 0 else 'S' if lat_or_lon == 'lat' else 'E' if lat_or_lon == 'lon' and decimal >= 0 else 'W'
        
        return f"{abs(degrees)}°{minutes}'{seconds:.2f}\"{direction}"
    
    def _get_existing_classes(self):
        """Get list of existing class directories"""
        output_dir = self.config['data_collection']['output_directory']
        if os.path.exists(output_dir):
            return [d for d in os.listdir(output_dir) 
                   if os.path.isdir(os.path.join(output_dir, d))]
        return []
    
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
        
        # Calculate audio level
        level = np.abs(indata).mean()
        self.level_label.setText(f"Audio Level: {20 * np.log10(level + 1e-10):.1f} dB")
        
        if self.is_recording:
            self.audio_buffer.extend(indata[:, 0])
            
            # Check for silence
            if level < self.silence_threshold:
                if self.silence_start is None:
                    self.silence_start = time.currentTime
            else:
                self.silence_start = None
            
            # Update progress
            progress = min(100, len(self.audio_buffer) / (self.sample_rate * self.recording_duration) * 100)
            self.progress_bar.setValue(int(progress))
    
    def toggle_recording(self):
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        self.stream.start()
        self.is_recording = True
        self.record_button.setText("Stop Recording")
        self.status_label.setText("Status: Recording")
        self.audio_buffer = []
        self.silence_start = None
        self.progress_bar.setValue(0)
    
    def stop_recording(self):
        self.stream.stop()
        self.is_recording = False
        self.record_button.setText("Start Recording")
        self.status_label.setText("Status: Processing...")
        
        if len(self.audio_buffer) > 0:
            self.save_recording()
        
        self.status_label.setText("Status: Ready")
        self.progress_bar.setValue(0)
    
    def save_recording(self):
        # Convert buffer to numpy array
        audio_data = np.array(self.audio_buffer)
        
        # Normalize audio
        audio_data = librosa.util.normalize(audio_data)
        
        # Create class directory if it doesn't exist
        class_name = self.class_selector.currentText()
        class_dir = os.path.join(self.config['data_collection']['output_directory'], class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{class_name}_{timestamp}.{self.config['data_collection']['output_format']}"
        filepath = os.path.join(class_dir, filename)
        
        # Save audio file
        librosa.output.write_wav(filepath, audio_data, self.sample_rate)
        
        # Save metadata including location
        metadata = {
            'filename': filename,
            'class': class_name,
            'timestamp': timestamp,
            'sample_rate': self.sample_rate,
            'duration': len(audio_data) / self.sample_rate,
            'location': self.location_manager.get_current_location()
        }
        
        # Save metadata as JSON
        metadata_path = filepath.replace(self.config['data_collection']['output_format'], 'json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Update class selector if new class was added
        if class_name not in self._get_existing_classes():
            self.class_selector.addItem(class_name)

def main():
    app = QApplication([])
    window = DataCollector()
    window.show()
    app.exec()

if __name__ == "__main__":
    main() 