# Blood Cell Classification using Convolutional Neural Networks (CNN)
<p>In this project, I built, trained and evaluated a CNN model to classify white blood cell (Monocytes vs Lymphocyte) from Paul Mooney's <a href="https://www.kaggle.com/datasets/paultimothymooney/blood-cells"> blood cell images </a> from Kaggle</p>

## Project Summary
<ul>
<li>I worked with 1000 blood cell image</li>
<li>I used Python 3; Tensorflow and Keras on Google Collab</li>
<li>The model was trained with a binary image classifier</li>
<li>Image size used was 128 x 128</li>
<li>Batch size was 32</li>
</ul>

## CNN Architecture
A lightweight Sequential CNN:
- Rescaling layer (normalize pixels to [0, 1])
- 3 Convolutional blocks (32 → 64 → 128 filters) with ReLU activation and MaxPooling
- Flatten + Dense layers
- Output: Single neuron with sigmoid activation (binary classification)

## Training & Challenges
**Initial attempt**  
The model was stuck at ~50% accuracy (random guessing) for 5 epochs.  
**Root causes identified (Thanks to Gemini AI)**:
- Improper handling of normalization (separate `.map()` + internal Rescaling)
- Using logits output with mismatched loss function

**Fix applied**:
- Moved Rescaling inside the model
- Switched to sigmoid activation + standard `binary_crossentropy`
- Retrained for 10 epochs

## Results
After the fix:
- Training accuracy improved from ~49% → **88.8%**
- Training loss decreased from ~0.73 → **0.25**
- Both Accuracy and Loss were visualized

## Predictions
**The model predicted accurately on images from the dataset, however, with images from the internet, prediction was inaccurate**



