from tkinter import Tk, filedialog, Button, Label, messagebox, OptionMenu, StringVar
import cv2
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import shutil
import time

# 共享的全局变量和函数
resize_option = "640x480"  # 預設影像大小選項

def select_video_file():
    root = Tk()
    root.withdraw()
    video_path = filedialog.askopenfilename(title="Select a video file")
    return video_path

def end_program():
    global root  # 使用全局变量
    # Remove temporary folder and its contents
    print("Deleting 'temp' folder...")
    try:
        shutil.rmtree("temp")
        print("'temp' folder deleted.")
    except FileNotFoundError:
        print("'temp' folder not found.")
    # Exit the program
    root.quit()

# AE测试相关的函数和变量
def estimate_brightness(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray_frame)
    return brightness

def extract_frames_ae(video_path, output_folder, selected_size):
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
    
    # Initialize lists to store brightness values and time in seconds
    brightness_values = []
    times = []
    
    # Determine the target frame size based on selected size
    if selected_size == "HD":
        target_size = (1920, 1080)
    elif selected_size == "640x480":
        target_size = (640, 480)
    else:  # Default to 720P
        target_size = (1280, 720)
    
    # Start capturing frames
    frame_count = 0
    progress_bar = tqdm(total=total_frames, desc="Extracting Frames", unit="frames")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Calculate brightness of the frame
        brightness = estimate_brightness(frame)
        brightness_values.append(brightness)
        
        # Calculate time in seconds
        time_sec = frame_count / fps
        times.append(time_sec)
        
        # Resize the frame
        frame = cv2.resize(frame, target_size)
        
        # Save the frame as image
        frame_filename = f"{output_folder}/frame_{frame_count}.jpg"
        cv2.imwrite(frame_filename, frame)
        
        frame_count += 1
        progress_bar.update(1)
    
    # Release the video capture object
    cap.release()
    progress_bar.close()
    print(f"Frames extracted: {frame_count}/{total_frames}")
    
    # Plot brightness values
    plt.plot(times, brightness_values)
    plt.title('AE_Test')
    plt.xlabel('Frame')
    plt.ylabel('Brightness')
    plt.grid(True)
    plt.savefig(f"AE_Test.png")
    plt.show()
    plt.close()

def execute_program_ae():
    global resize_variable  # 使用全局变量
    selected_size = resize_variable.get()
    
    # Prompt user to select a video file
    video_path = select_video_file()
    if video_path:
        # Set output folder in temp directory
        output_folder = "temp"

        # Show message box indicating execution
        messagebox.showinfo("Executing", "Please click the button and Executing AE program. Please wait for 1 mins...")

        # Call the function with user input
        extract_frames_ae(video_path, output_folder, selected_size)

        # Close the info message box after 2 seconds
        messagebox.showinfo("Done", "AE Video processing completed.")

# AWB测试相关的函数和变量
def estimate_color_temperature(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    avg_a = np.mean(img_lab[:, :, 1])
    avg_b = np.mean(img_lab[:, :, 2])
    color_temperature = None
    if avg_a > 1:
        color_temperature = 6500 + ((avg_b - 128) * 59)
    else:
        color_temperature = 6500 - ((avg_b - 128) * 59)
    return color_temperature

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

    # plt.figure(figsize=(10, 6))
    plt.plot(times, color_temp_values)
    plt.title('AWB_Test')
    plt.xlabel('Frame')
    plt.ylabel('Color Temperature (K)')
    plt.grid(True)
    plt.savefig('AWB_Test.png')
    plt.show()


def extract_frames_awb(video_path, output_folder, selected_size):
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
        
        # Determine the target frame size based on selected size
        if selected_size == "HD":
            target_size = (1920, 1080)
        elif selected_size == "640x480":
            target_size = (640, 480)
        else:  # Default to 720P
            target_size = (1280, 720)

        # Resize the frame
        frame = cv2.resize(frame, target_size)

        # Save the frame as image
        frame_filename = f"{output_folder}/frame_{frame_count}.jpg"
        cv2.imwrite(frame_filename, frame)

        frame_count += 1
        progress_bar.update(1)

    # Release the video capture object
    cap.release()
    progress_bar.close()

def execute_program_awb():
    global resize_variable  # 使用全局变量
    selected_size = resize_variable.get()
    
    # Prompt user to select a video file
    video_path = select_video_file()
    if video_path:
        # Set output folder in temp directory
        output_folder = "temp"

        # Show message box indicating execution
        messagebox.showinfo("Executing", "Please click the button and Executing AWB program. Please wait for 1 mins...")

        # Call the function with user input
        extract_frames_awb(video_path, output_folder, selected_size)

        # Calculate color temperatures
        color_temp_values = calculate_color_temperatures(output_folder)

        # Plot color temperature over time
        plot_color_temperature_over_time(color_temp_values, fps=None)

        # Close the info message box after 2 seconds
        messagebox.showinfo("Done", "AWB Video processing completed.")

# 共享的功能和主函数
def main():
    global root, resize_variable  # 使用全局变量
    # Create Tkinter window
    root = Tk()
    root.geometry("600x200")
    root.title("Video AE & AWB TEST")

    # Label for step 1
    step1_label = Label(root, text="Step 1")
    step1_label.place(relx=0.1,rely=0.1)

    # Dropdown menu for resize options
    resize_options = ["HD", "720P", "640x480"]
    resize_variable = StringVar(root)
    resize_variable.set(resize_option)  # Default option
    resize_dropdown = OptionMenu(root, resize_variable, *resize_options)
    resize_dropdown.place(relx=0.1,rely=0.2)

    # Label for step 2
    step2_label = Label(root, text="Step 2")
    step2_label.place(relx=0.3,rely=0.1)
    
    # Button to select AE video
    ae_button = Button(root, text="AE Test", command=execute_program_ae)
    ae_button.place(relx=0.3,rely=0.2)

    # Button to select AWB video
    awb_button = Button(root, text="AWB Test", command=execute_program_awb)
    awb_button.place(relx=0.3,rely=0.4)

    # Label for step 3
    step3_label = Label(root, text="Step 3")
    step3_label.place(relx=0.5,rely=0.1)

    # Button to end the program
    end_button = Button(root, text="Close the Program", command=end_program)
    end_button.place(relx=0.5,rely=0.2)

    root.mainloop()

if __name__ == "__main__":
    main()
