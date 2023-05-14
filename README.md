# FacialExpressionRecognition

- This is a sample project which **predicts** the ***emotion*** of a person in **Realtime**.

**facialExpressionRecognition.ipynb**
- Model is built using CNN with 4 convolution blocks having Convolution layer, MaxPooling2D, BatchNormalization and Dropout with activalion as 'ReLu'
- Then the model is flattened and passed on to Fully Connected layer before giving the output (prediction for emotion).
- Model is saved in json format, and the weigths have been saved as 'model.h5'

**model.py**
- Python file which predicts the expression of the input(video/webcam).
- Uses the model.json and model.h5 to give a probability for each expression.
- EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprised"]
- Then select the probability with maximum value and gives the output as the predicted Emotion.

**webcam_prediction**
- Uses computer vision library (open-cv) to open the webcam, identify face.
- Displays the emotion above the rectangular box, which displays face like below.

![image](https://github.com/soubhagyardas/FacialExpressionRecognition/assets/47771334/2546b9f7-09f0-42ba-9363-1a41e5fc8fe3)
