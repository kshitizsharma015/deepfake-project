import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import Xception
import cv2
import numpy as np
import sys
import json

# Step 1: Load pre-trained XceptionNet for spatial features
base_model = Xception(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
base_model.trainable = False # Freeze for transfer learning

# Step 2: Add Temporal CNN
def create_model():
    inputs = layers.Input(shape=(10, 299, 299, 3)) 
    x = layers.TimeDistributed(base_model)(inputs)
    x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)
    x = layers.Conv1D(128, kernel_size=3, activation='relu')(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Step 3: Preprocess video
def extract_frames(video_path, num_frames=10):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < num_frames and cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (299, 299))
            frames.append(frame / 255.0) # Normalize
    cap.release()
    
    # Handle short videos by duplicating the last frame
    if frames and len(frames) < num_frames:
        while len(frames) < num_frames:
            frames.append(frames[-1])
    elif not frames:
        return np.zeros((1, num_frames, 299, 299, 3))
        
    return np.array([frames])

# Step 4: Detect function
def detect_deepfake(video_path):
    # This function will be used after training
    # It needs a loaded model to work
    processed_frames = extract_frames(video_path)
    
    # The 'model' variable is loaded globally in the __main__ block
    prediction = model.predict(processed_frames)[0][0]
    
    # --- THIS IS THE FIX ---
    # We raise the threshold from 0.5 to 0.9
    # This makes the model less "suspicious" and
    # should fix the false positive problem with real videos.
    # You can tune this value (e.g., 0.85, 0.95)
    threshold = 0.9
    # ---------------------
    
    return {'fake_probability': float(prediction), 'is_fake': bool(prediction > threshold)}

# This part is for running the script directly for testing or from the server
if __name__ == "__main__":
    import sys
    import json
    import tensorflow as tf # Make sure tf is imported

    # We will wrap everything in a try...except block
    try:
        # server.js sends the video path as the 1st argument (index 1)
        # and the model path as the 2nd argument (index 2)
        video_file_path = sys.argv[1] 
        model_path = sys.argv[2]

        # Load the trained model using the path from the server
        model = tf.keras.models.load_model(model_path) 

        # Run the detection
        result = detect_deepfake(video_file_path)

        # Print the successful result as JSON
        print(json.dumps(result)) 
        
    except Exception as e:
        # If ANY error happens, catch it and print it as a VALID JSON object
        print(json.dumps({"error": str(e)}))
