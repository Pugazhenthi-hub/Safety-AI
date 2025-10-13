import os
import shutil

# Folder where all frames are currently saved
frame_folder = 'frames/'
# New folder where frames will be organized by class
organized_folder = 'frames_by_class/'

# List of all classes
classes = ['abuse', 'fire', 'explosion', 'anomaly_activity', 'normal']

# Create class folders inside frames_by_class
for cls in classes:
    os.makedirs(os.path.join(organized_folder, cls), exist_ok=True)

# Move frames into the correct class folder
for frame_name in os.listdir(frame_folder):
    for cls in classes:
        if cls in frame_name.lower():  # check if class name is in frame filename
            shutil.move(os.path.join(frame_folder, frame_name),
                        os.path.join(organized_folder, cls, frame_name))
            break

print("Frames organized by class successfully!")
