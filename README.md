# NoiseNinja - Sound Identification App

NoiseNinja is a real-time sound identification application that uses machine learning to identify sounds from your microphone input.

## Features

- Real-time sound capture and processing
- Deep learning-based sound classification
- User-friendly GUI interface
- Support for 527 different sound classes

## Requirements

- Python 3.8 or higher
- PyTorch
- CUDA-capable GPU (recommended but not required)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/NoiseNinja.git
cd NoiseNinja
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:
```bash
python sound_identifier.py
```

2. Click the "Start Recording" button to begin sound identification
3. The app will display the detected sound in real-time
4. Click "Stop Recording" to stop the identification process

## Model Training

The current version uses a pre-trained model based on the AudioSet dataset. To train your own model:

1. Download the AudioSet dataset
2. Use the provided model architecture in `model.py`
3. Train the model using PyTorch
4. Save the trained model weights

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 