import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from tkinter import Tk, filedialog
from tqdm import tqdm
import shutil
import time
timestr = time.strftime("%Y%m%d") #%Y%m%d%H%M%S

def extract_frames(video_path, output_folder):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open the video file.")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Initialize lists to store brightness values and time in seconds
    brightness_values = []
    times = []
    
    # Start capturing frames
    frame_count = 0
    progress_bar = tqdm(total=total_frames, desc="Extracting Frames", unit="frames")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate brightness of the frame
        brightness = np.mean(gray_frame)
        brightness_values.append(brightness)
        
        # Calculate time in seconds
        time_sec = frame_count / fps
        times.append(time_sec)
        
        # Save every frame as image (HD resolution)
        frame_filename = f"{output_folder}/frame_{frame_count}.jpg"
        resized_frame = cv2.resize(frame, (1280, 720))
        cv2.imwrite(frame_filename, resized_frame)
        
        frame_count += 1
        progress_bar.update(1)
    
    # Release the video capture object
    cap.release()
    progress_bar.close()
    print(f"Frames extracted: {frame_count}/{total_frames}")
    
    # Plot brightness values
    plt.figure(figsize=(10, 6))
    plt.plot(times, brightness_values)
    plt.title('AE_Test')
    plt.xlabel('Frame')
    plt.ylabel('Brightness')
    plt.grid(True)
    plt.savefig(f"AE_Test.png")
    plt.show()
    plt.close()

def select_video_file():
    root = Tk()
    root.withdraw()
    video_path = filedialog.askopenfilename(title="Select a video file")
    return video_path

def main():
    print("select when you're ready to select the video file.")
    
    # Prompt user to select a video file using a GUI
    time.sleep(1)
    video_path = select_video_file()
    
    # Set output folder in temp directory
    output_folder = "temp"
    
    # Call the function with user input
    extract_frames(video_path, output_folder)
    
    print("Delete 'temp' folder now.....")
    # Remove temporary folder and its contents
    shutil.rmtree("temp")
    
if __name__ == "__main__":
    main()
