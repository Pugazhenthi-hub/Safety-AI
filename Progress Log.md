## date:22-09-2025 ##

we have enrolled coursera course which is Introduction to Artificial Intelligence
created python code to our project 
It run successfully with a random output

## date:25-09-2025 ##

Today we have completed half module in Introduction to Artificial Intelligence which is enrolled in coursera,and discussed about our project.

## date:09-10-2025 ##

created phase-1 :project overview and objectives for our project and the file is committed in git.


## date:13-10-2025 ##

## Current Project Status Update:
We have completed the data preparation stage of the project.
Collected and organized violence and normal activity videos (including abuse, fire, explosion, anomaly activity, and normal videos).
Extracted video frames and grouped them by class inside the frames_by_class folder.
Set up the environment with OpenCV, NumPy, PyTorch, and other necessary Python libraries.
Designed the plan for the CNN+LSTM model architecture, which will process spatial (CNN) and temporal (LSTM) features.
The next step is to train and evaluate the CNN+LSTM model using the prepared frame sequences.

## ✅ Work Completed

*. Dataset collection and preprocessing
*. Frame extraction and organization
*. Data pipeline design for sequence generation
*. Model architecture planning (CNN + LSTM)

## 🔜 Next Step

*. Implement and train the CNN+LSTM model on the prepared dataset.
*. Evaluate model performance and fine-tune accuracy.
I have included the dataset files 


## Date:16-10-2025

1.Reviewed the dataset and checked its structure.

2.Attempted to prepare sequences for CNN+LSTM model training, but encountered memory allocation issues due to the dataset size.

3.Explored solutions for handling large datasets efficiently (like reducing batch size, using generators, or resizing input data).

4.Planned the next steps: either optimize the preprocessing code or use a smaller subset of data to proceed with model training.

## date:30-10-2025

Today, we continued working on our AI project (CNN + LSTM model). We checked the dataset, prepared the data for training, and identified a memory issue during sequence preparation. We discussed possible solutions to handle large datasets more efficiently and planned to optimize the preprocessing in the next session.

## Date: 03-11-2025

Violence Detection using CNN + LSTM

We have successfully completed the data collection and preprocessing stages of our project. We gathered video datasets covering various categories such as abuse, fire, explosion, anomaly activity, and normal scenes. These videos were carefully processed by extracting individual frames using OpenCV.

The extracted frames have been organized class-wise inside a structured folder system (frames_by_class/), which allows our model to easily differentiate between different types of activities. This step ensures that both violent and non-violent scenarios are well-represented for training.

Currently, we are in the model training phase. We are training a hybrid CNN + LSTM model:

The CNN (Convolutional Neural Network) part extracts spatial features from each video frame (like shapes, movements, and visual patterns).

The LSTM (Long Short-Term Memory) part learns the temporal sequence — how frames change over time — to understand the flow of actions.

The combined CNN+LSTM architecture is designed to detect and classify violent versus non-violent activities effectively. We are monitoring the model’s accuracy and performance during training and will soon proceed to the evaluation stage to test its effectiveness on unseen data.

## Date: 06-11-2025

🧠 AI Project Status Update — CNN + LSTM Model

Current Progress:

The CNN + LSTM model for video-based activity recognition has been successfully trained using the prepared dataset.

Training completed locally with high performance — Test Accuracy: 99.8 %.

The trained model has been saved as cnn_lstm_model.pth (≈ 205 MB).

We have also prepared a testing script (test_model.py) to evaluate the model on new or unseen videos.

The project structure is organized with:

Dataset split into class folders (frames_by_class)

Preprocessing scripts (extract_frames.py, organize_frames.py, prepare_sequences.py)

Training script (train_cnn_lstm.py)

Testing script (test_model.py)

Next Steps:

Test the model on new video samples to validate real-world accuracy.

Optionally integrate real-time webcam/video stream prediction.

Prepare final documentation and results report.