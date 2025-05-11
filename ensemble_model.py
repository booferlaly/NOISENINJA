import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import torch
import torch.nn as nn
from model import SoundClassifier
from post_processor import PredictionPostProcessor, ClassSpecificThresholds

class EnsembleSoundClassifier:
    def __init__(self, cnn_model_path=None, confidence_threshold=0.5):
        # Initialize YAMNet
        self.yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
        self.yamnet_classes = self.yamnet_model.class_names
        
        # Initialize VGGish
        self.vggish_model = hub.load('https://tfhub.dev/google/vggish/1')
        
        # Initialize CNN model
        self.cnn_model = load_model(cnn_model_path)
        
        # Initialize post-processor
        self.post_processor = PredictionPostProcessor(
            window_size=5,
            min_confidence=confidence_threshold,
            noise_threshold=0.3
        )
        
        # Initialize class-specific thresholds
        self.class_thresholds = ClassSpecificThresholds()
        
        # Set confidence threshold
        self.confidence_threshold = confidence_threshold
        
        # Create mapping between models
        self.class_mapping = self._create_class_mapping()
        
        # Initialize model weights
        self.model_weights = {
            'yamnet': 0.4,
            'vggish': 0.3,
            'cnn': 0.3
        }
    
    def _create_class_mapping(self):
        """
        Create a mapping between different model class indices
        """
        mapping = {}
        for i, yamnet_class in enumerate(self.yamnet_classes):
            # Find the closest matching class in other models
            # This is a placeholder - implement proper mapping
            mapping[i] = {
                'vggish': i % 128,  # VGGish has 128 classes
                'cnn': i % 527  # CNN has 527 classes
            }
        return mapping
    
    def predict(self, audio_data, sample_rate):
        """
        Make predictions using all models and combine results
        """
        # Apply spectral filtering
        filtered_audio = self.post_processor.apply_spectral_filtering(audio_data, sample_rate)
        
        # YAMNet prediction
        scores, embeddings, spectrogram = self.yamnet_model(filtered_audio)
        yamnet_scores = scores.numpy().mean(axis=0)
        yamnet_pred = np.argmax(yamnet_scores)
        yamnet_conf = yamnet_scores[yamnet_pred]
        
        # VGGish prediction
        vggish_embeddings = self.vggish_model(filtered_audio)
        vggish_scores = tf.nn.softmax(vggish_embeddings).numpy().mean(axis=0)
        vggish_pred = np.argmax(vggish_scores)
        vggish_conf = vggish_scores[vggish_pred]
        
        # CNN prediction
        mel_spec = self._create_mel_spectrogram(filtered_audio, sample_rate)
        cnn_pred, cnn_conf = predict_sound(self.cnn_model, mel_spec)
        
        # Combine predictions with weighted voting
        predictions = {
            'yamnet': (yamnet_pred, yamnet_conf),
            'vggish': (vggish_pred, vggish_conf),
            'cnn': (cnn_pred, cnn_conf)
        }
        
        # Apply model weights
        weighted_scores = {}
        for model_name, (pred, conf) in predictions.items():
            weight = self.model_weights[model_name]
            if pred not in weighted_scores:
                weighted_scores[pred] = 0
            weighted_scores[pred] += conf * weight
        
        # Get final prediction
        if weighted_scores:
            final_pred = max(weighted_scores.items(), key=lambda x: x[1])
            final_pred_class = final_pred[0]
            final_conf = final_pred[1]
        else:
            return "Unknown", 0.0
        
        # Get class name
        if final_pred_class < len(self.yamnet_classes):
            class_name = self.yamnet_classes[final_pred_class]
        else:
            class_name = f"Class_{final_pred_class}"
        
        # Apply post-processing
        processed_pred, processed_conf = self.post_processor.process(class_name, final_conf)
        
        # Check class-specific threshold
        if not self.class_thresholds.should_accept_prediction(processed_pred, processed_conf):
            return "Unknown", 0.0
        
        return processed_pred, processed_conf
    
    def _create_mel_spectrogram(self, audio_data, sample_rate):
        """
        Create mel spectrogram for CNN model
        """
        import librosa
        mel_spec = librosa.feature.melspectrogram(
            y=audio_data,
            sr=sample_rate,
            n_mels=128,
            fmax=8000
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db
    
    def set_model_weights(self, yamnet_weight, vggish_weight, cnn_weight):
        """
        Set the weights for each model in the ensemble
        """
        total = yamnet_weight + vggish_weight + cnn_weight
        self.model_weights = {
            'yamnet': yamnet_weight / total,
            'vggish': vggish_weight / total,
            'cnn': cnn_weight / total
        }
    
    def set_class_threshold(self, class_name, threshold):
        """
        Set a specific confidence threshold for a class
        """
        self.class_thresholds.set_threshold(class_name, threshold)

def load_model(model_path=None):
    """
    Load the CNN model
    """
    model = SoundClassifier()
    if model_path and torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path))
        model = model.cuda()
    elif model_path:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model 