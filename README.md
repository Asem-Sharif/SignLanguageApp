# Sign Language Application
An application that facilitates the creation of a sign language dataset, saving it, and training/testing a `TensorFlow.Keras` model on the captured data. 
-
This project integrates a real-time feed for hand detection and sign classification features.
-
---
## Features
The app consists of six main sections:
### 1. Camera Frame (`Main.main_canvas`)
- Displays live video feed from the `OpenCV.VideoCapturer`.
- **Right or Left Click:** Toggles the `Main.shy` property. If `True`, the camera is enabled but the frame is not displayed.
### 2. Detector Frame (`Main.side_canvas`)
- Displays results from the `Mediapipe` hand detector if a hand is detected.
- **Right Click:** Displays `Main.landmarks_on_blank` image.
- **Left-Hold Click:** Displays raw RGB image.
- **Scroll:** Applies various preprocessing functions.
### 3. Control Frame (`Main.control_section`)
- Provides controls to edit camera and detector parameters.
### 4. Model Frame (`Main.model_section`)
- Allows initialization of a the CNN model.
- Facilitates training and testing with live updates of accuracy and loss metrics.
- Allows other model evaluation metrics like precision, recall, f1 score, roc-auc, and confusion matrix.
### 5. Terminal Frame (`Main.terminal_section`)
- Displays important information and status updates.
### 6. Data Frame (`Main.data_section`)
- Enables the creation or import of datasets.
---
## Requirements
Ensure that all required Python packages are installed before running the app. Use the `REQUIREMENTS.txt` file to install dependencies:
```bash
pip install -r REQUIREMENTS.txt
```
---
## Warning:
- Ensure that the `control_section > detector_setting > method > output_shape` is `square` or `border` to avoid shape-related errors while training the model.
- Ensure that the `control_section > detector_setting > method > output_size` should not be too large to avoid resource exhaustion errors if the data is not properly preprocessed before training.
- Removing `CNN` from `App.py` will lead to pickle import errors whenever the `CNN` instance is initialized in a main module, such as in `Colab` or `Jupyter` notebooks, or in any other module than `Utils.Models.CNN`.
```python
from Main import App, CNN
```
---
## How to Use
1. Start by running the application `App.py`.
2. Enable the camera then and detector.
3. Generate the dataset and capture the signs.
4. Export the dataset, set the label to 'Label'.
5. Generate a model step by step, ensuring that the parameters are well-tuned.
6. Train and evaluate the model, monitoring the loss and accuracy.
7. Activate the model and begin displaying similar signs.
8. The prediction and confidence should update once every 10-30 frames.
9. Save the model for later usage.
---
## Author
**Asem Al-Sharif, December 2024.**
