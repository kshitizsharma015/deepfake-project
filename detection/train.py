import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# Import from our new, upgraded file
from detect import create_model, extract_frames 
# --- IMPORT NEW CALLBACKS ---
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# --- 1. LOAD DATA FROM METADATA FILE ---
print("Loading data from metadata file...")
METADATA_FILE = 'New_DF.csv' 
VIDEO_SOURCE_FOLDERS = ['dfdc_train_part_49', 'Modified Dataset']

try:
    df = pd.read_csv(METADATA_FILE)
except FileNotFoundError:
    print(f"Error: Could not find '{METADATA_FILE}'. Make sure it's in the 'detection' folder.")
    exit()

real_vids = []
fake_vids = []

for index, row in df.iterrows():
    filename = row['video']
    label = row['label']
    
    source_file_path = None
    for folder in VIDEO_SOURCE_FOLDERS:
        potential_path = os.path.join(folder, filename)
        if os.path.exists(potential_path):
            source_file_path = potential_path
            break
    
    if source_file_path:
        if label.upper() == 'REAL':
            real_vids.append(source_file_path)
        elif label.upper() == 'FAKE':
            fake_vids.append(source_file_path)

print(f"Found {len(real_vids)} REAL videos and {len(fake_vids)} FAKE videos.")

# --- 2. PREPARE DATA FOR TRAINING ---
videos = real_vids + fake_vids
labels = [0] * len(real_vids) + [1] * len(fake_vids)

# --- TRAINING UPGRADE (More Data) ---
# We will use 2000 videos now instead of 200.
# You can increase this number for even better results.
# Make sure you have at least 2000 total videos (e.g., 1000 real, 1000 fake)
TRAIN_SET_SIZE = 2000 
if len(videos) < TRAIN_SET_SIZE:
    TRAIN_SET_SIZE = len(videos) # Use all videos if we have less than 2000
    print(f"Warning: Dataset is smaller than 2000. Using all {len(videos)} videos.")

subset_videos, _, subset_labels, _ = train_test_split(videos, labels, train_size=TRAIN_SET_SIZE, stratify=labels, random_state=42)

# We now split into training AND validation sets
# The model will check its performance on the validation set during training
X_train, X_val, y_train, y_val = train_test_split(subset_videos, subset_labels, test_size=0.2, stratify=subset_labels, random_state=42)
print(f"Training with {len(X_train)} videos, validating with {len(X_val)} videos.")


# --- 3. DATA GENERATOR AND MODEL TRAINING ---
def data_gen(vids, lbls, batch_size=4):
    while True:
        # Shuffle the data each epoch
        indices = np.arange(len(vids))
        np.random.shuffle(indices)
        
        for i in range(0, len(vids), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_vids = [vids[idx] for idx in batch_indices]
            batch_lbls = [lbls[idx] for idx in batch_indices]
            
            batch_frames = [extract_frames(v)[0] for v in batch_vids]
            
            yield np.array(batch_frames), np.array(batch_lbls)

model = create_model()

# Create generators for both training and validation
# Use a larger batch size for faster training
BATCH_SIZE = 8
train_generator = data_gen(X_train, y_train, batch_size=BATCH_SIZE)
val_generator = data_gen(X_val, y_val, batch_size=BATCH_SIZE)

# --- TRAINING UPGRADE (Smarter Training) ---
# These callbacks will make the training more efficient.
# 1. Stop training if the model isn't improving
early_stopping = EarlyStopping(
    monitor='val_loss', # Stop when validation loss stops decreasing
    patience=5,         # Wait 5 epochs before stopping
    verbose=1,
    restore_best_weights=True # Keep the best version of the model
)

# 2. Reduce the learning rate if training gets stuck
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2, # Reduce LR by 80%
    patience=2,
    verbose=1,
    min_lr=1e-6 # Don't let it get too small
)

print("\nStarting model training...")
# We add the validation data and callbacks to the fit() call
model.fit(
    train_generator,
    steps_per_epoch=max(1, len(X_train) // BATCH_SIZE),
    epochs=30, # Train for more epochs (EarlyStopping will find the best one)
    validation_data=val_generator,
    validation_steps=max(1, len(X_val) // BATCH_SIZE),
    callbacks=[early_stopping, reduce_lr] # Add our new callbacks
)
print("Training complete.")

# Save the new, better model with a new name
model.save('deepfake_model_v2.h5')
print("Model saved to deepfake_model_v2.h5")
