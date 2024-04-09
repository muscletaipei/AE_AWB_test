from tkinter import Tk, filedialog, Button, Label, messagebox, OptionMenu, StringVar
import cv2
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import shutil
import time

timestr = time.strftime("%Y%m%d")  # %Y%m%d%H%M%S
root = None  # 全局變數
resize_option = "720P"  # 預設影像大小選項

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
    
    # Determine the target frame size
    if resize_option == "HD":
        target_size = (1920, 1080)
    elif resize_option == "640X480":
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
        
        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate brightness of the frame
        brightness = np.mean(gray_frame)
        brightness_values.append(brightness)
        
        # Calculate time in seconds
        time_sec = frame_count / fps
        times.append(time_sec)
        
        # Resize the frame
        resized_frame = cv2.resize(frame, target_size)
        
        # Save the frame as image
        frame_filename = f"{output_folder}/frame_{frame_count}.jpg"
        cv2.imwrite(frame_filename, resized_frame)
        
        frame_count += 1
        progress_bar.update(1)
    
    # Release the video capture object
    cap.release()
    progress_bar.close()
    print(f"Frames extracted: {frame_count}/{total_frames}")
    
    # Plot brightness values
    plt.plot(times, brightness_values)
    plt.title('AE_Test')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Brightness')
    plt.savefig(f"AE_Test.png")
    plt.show()
    plt.close()

def select_video_file():
    root = Tk()
    root.withdraw()
    video_path = filedialog.askopenfilename(title="Select a video file")
    return video_path

def execute_program():
    # Prompt user to select a video file
    video_path = select_video_file()
    if video_path:
        # Set output folder in temp directory
        output_folder = "temp"

        # Show message box indicating execution
        global messagebox_id
        messagebox_id = messagebox.showinfo("Executing", "Please click the button and Executing program. Please wait for 1 mins...")

        # Call the function with user input
        extract_frames(video_path, output_folder)

        # Close the info message box after 2 seconds
        root.after(2000, close_messagebox)

def close_messagebox():
    global messagebox_id
    if messagebox_id:
        root.after_cancel(messagebox_id)
        messagebox_id = None

def end_program():
    global root  # 使用全局變數
    # Remove temporary folder and its contents
    print("Deleting 'temp' folder...")
    shutil.rmtree("temp")
    print("'temp' folder deleted.")
    # Exit the program
    root.quit()

def change_resize_option(option):
    global resize_option
    resize_option = option

def main():
    global root  # 使用全局變數
    # Create Tkinter window
    root = Tk()
    root.geometry("600x400")
    root.title("Video AE TEST")

    # Dropdown menu for resize options
    resize_options = ["HD", "640X480", "720P"]
    resize_variable = StringVar(root)
    resize_variable.set(resize_option)  # Default option
    resize_dropdown = OptionMenu(root, resize_variable, *resize_options, command=change_resize_option)
    resize_dropdown.place(relx=0.1,rely=0.2)

    # Label for step 1
    step1_label = Label(root, text="Step 1")
    step1_label.place(relx=0.1,rely=0.1)

    # Button to select video
    execute_button = Button(root, text="Select video", command=execute_program)
    execute_button.place(relx=0.3,rely=0.2)

    # Label for step 2
    step2_label = Label(root, text="Step 2")
    step2_label.place(relx=0.3,rely=0.1)

    # Button to end the program
    end_button = Button(root, text="Close Program", command=end_program)
    end_button.place(relx=0.5,rely=0.2)

    # Label for step 3
    step3_label = Label(root, text="Step 3")
    step3_label.place(relx=0.5,rely=0.1)


    root.mainloop()

if __name__ == "__main__":
    main()
