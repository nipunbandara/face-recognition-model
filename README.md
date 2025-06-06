# Face Recognition Model

## Installation

```bash
git clone https://github.com/yourusername/face-recognition-model.git
cd face-recognition-model
pip install -r requirements.txt
```

## Model Creation Instructions

### 1. Live Capture

To capture images from your webcam for training:

```bash
python capture_dataset.py
```
Follow the prompts to save images for each person.

### 2. Training the Model

Train the face recognition model using the captured images:

```bash
python train_model.py
```

### 3. Testing and Live Recognition

To run live face recognition using your webcam:

```bash
python recognize_live.py
```

### 4. Running the Main Script

Alternatively, you can use the main script to perform all steps:
Additionally, with a TensorFlow.js format conversion function for h5 keras model

```bash
python main.py
```