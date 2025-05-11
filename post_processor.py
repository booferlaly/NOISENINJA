import numpy as np
from collections import deque
import scipy.signal as signal

class PredictionPostProcessor:
    def __init__(self, window_size=5, min_confidence=0.5, noise_threshold=0.3):
        self.window_size = window_size
        self.min_confidence = min_confidence
        self.noise_threshold = noise_threshold
        
        # Circular buffer for temporal smoothing
        self.prediction_buffer = deque(maxlen=window_size)
        self.confidence_buffer = deque(maxlen=window_size)
        
        # Noise filtering parameters
        self.smoothing_factor = 0.3
        self.last_valid_prediction = None
        self.last_valid_confidence = 0.0
    
    def process(self, prediction, confidence):
        """
        Process a single prediction with temporal smoothing and noise filtering
        Args:
            prediction: str, the predicted class
            confidence: float, the confidence score
        Returns:
            processed_prediction: str, the processed prediction
            processed_confidence: float, the processed confidence
        """
        # Add current prediction to buffer
        self.prediction_buffer.append(prediction)
        self.confidence_buffer.append(confidence)
        
        # Apply temporal smoothing
        smoothed_prediction = self._temporal_smoothing()
        smoothed_confidence = np.mean(self.confidence_buffer)
        
        # Apply noise filtering
        if smoothed_confidence < self.noise_threshold:
            # If confidence is too low, maintain last valid prediction
            if self.last_valid_prediction is not None:
                return self.last_valid_prediction, self.last_valid_confidence
            else:
                return "Unknown", 0.0
        
        # Update last valid prediction
        if smoothed_confidence >= self.min_confidence:
            self.last_valid_prediction = smoothed_prediction
            self.last_valid_confidence = smoothed_confidence
        
        return smoothed_prediction, smoothed_confidence
    
    def _temporal_smoothing(self):
        """
        Apply temporal smoothing to predictions using a sliding window
        """
        if len(self.prediction_buffer) < self.window_size:
            return self.prediction_buffer[-1]
        
        # Count occurrences of each prediction in the window
        prediction_counts = {}
        for pred in self.prediction_buffer:
            prediction_counts[pred] = prediction_counts.get(pred, 0) + 1
        
        # Return the most common prediction
        return max(prediction_counts.items(), key=lambda x: x[1])[0]
    
    def apply_spectral_filtering(self, audio_data, sample_rate):
        """
        Apply spectral filtering to remove noise
        Args:
            audio_data: numpy array of audio data
            sample_rate: int, the sample rate
        Returns:
            filtered_audio: numpy array of filtered audio data
        """
        # Design a bandpass filter
        nyquist = sample_rate / 2
        low = 20 / nyquist
        high = 8000 / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        
        # Apply the filter
        filtered_audio = signal.filtfilt(b, a, audio_data)
        
        return filtered_audio
    
    def reset(self):
        """
        Reset the post-processor state
        """
        self.prediction_buffer.clear()
        self.confidence_buffer.clear()
        self.last_valid_prediction = None
        self.last_valid_confidence = 0.0

class ClassSpecificThresholds:
    def __init__(self):
        self.thresholds = {}
        self.default_threshold = 0.5
    
    def set_threshold(self, class_name, threshold):
        """
        Set a specific confidence threshold for a class
        """
        self.thresholds[class_name] = threshold
    
    def get_threshold(self, class_name):
        """
        Get the confidence threshold for a class
        """
        return self.thresholds.get(class_name, self.default_threshold)
    
    def should_accept_prediction(self, class_name, confidence):
        """
        Check if a prediction should be accepted based on class-specific threshold
        """
        threshold = self.get_threshold(class_name)
        return confidence >= threshold 