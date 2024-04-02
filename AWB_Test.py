import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from tqdm import tqdm

from tkinter import Tk, filedialog, Label
import time

def estimate_color_temperature(frame):
    # 將BGR轉換為RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 將RGB轉換為LAB色彩空間
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    
    # 計算色溫
    avg_a = np.mean(img_lab[:, :, 1])
    avg_b = np.mean(img_lab[:, :, 2])
    
    color_temperature = None
    
    if avg_a > 1:
        color_temperature = 6500 + ((avg_b - 128) * 59)
    else:
        color_temperature = 6500 - ((avg_b - 128) * 59)
    
    return color_temperature

def extract_frames(video_path, output_folder):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open the video file.")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Start capturing frames
    frame_count = 0
    progress_bar = tqdm(total=total_frames, desc="Extracting Frames", unit="frames")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Save the frame as image
        frame = cv2.resize(frame, (1280, 720))
        # Save the frame as image
        frame_filename = f"{output_folder}/frame_{frame_count}.jpg"
        cv2.imwrite(frame_filename, frame)

        frame_count += 1
        progress_bar.update(1)

    # Release the video capture object
    cap.release()
    progress_bar.close()

def calculate_color_temperatures(folder):
    color_temp_values = []
    files = [file for file in os.listdir(folder) if file.endswith('.jpg')]
    
    with tqdm(total=len(files), desc="Calculating Color Temperatures") as pbar:
        for filename in files:
            # Read the image
            image_path = os.path.join(folder, filename)
            img = cv2.imread(image_path)

            # Calculate color temperature
            color_temperature = estimate_color_temperature(img)
            color_temp_values.append(color_temperature)
            pbar.update(1)

    return color_temp_values

def plot_color_temperature_over_time(color_temp_values, fps):
    total_frames = len(color_temp_values)
    times = np.arange(0, total_frames)

    plt.figure(figsize=(10, 6))
    plt.plot(times, color_temp_values)
    plt.title('AWB_Test')
    plt.xlabel('Frame')
    plt.ylabel('Color Temperature (K)')
    # plt.xticks(np.arange(0, total_frames + 1, step=100))
    #plt.yticks(np.arange(5000, 12001, 1000))
    plt.grid(True)
    plt.savefig('AWB_Test.png')
    plt.show()

def select_video_file():
    root = Tk()
    root.withdraw()
    video_path = filedialog.askopenfilename(title="Select a video file")
    return video_path

def main():
    print("Select when you're ready to select the video file.")

    # Prompt user to select a video file using a GUI
    video_path = select_video_file()

    # Output folder for frames
    output_folder = "temp"

    # Call the function to extract frames
    extract_frames(video_path, output_folder)

    # Calculate color temperatures
    color_temp_values = calculate_color_temperatures(output_folder)

    # Get video properties
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Plot color temperature over time
    plot_color_temperature_over_time(color_temp_values, fps)

    # Clean up: Remove temporary frames
    print("Delete folder 'temp' now")
    shutil.rmtree(output_folder)
    print("Temporary folder 'temp' deleted.")
    print("The test completed..")

if __name__ == "__main__":
    main()
