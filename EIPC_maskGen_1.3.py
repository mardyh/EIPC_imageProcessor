#--------------------------------------#
#--------------IMPORTS-----------------#
#--------------------------------------#
import os
import sys
from pathlib import Path
from tkinter import Tk, Button, Label, Checkbutton, IntVar, filedialog, messagebox, ttk, Frame, StringVar, Radiobutton, Entry
from rembg import remove, new_session
from PIL import Image, ImageOps
import io
import cv2
import math
import subprocess
import shutil
import numpy as np

#--------------------------------------#
#-----------------VARS-----------------#
#--------------------------------------#
custom_model_path = None  # Will store user-selected model path if applicable
selected_generate_folder = None
selected_combine_folder = None

total_steps = 0
current_progress = 0

#----------------------------------------#
#--------------FUNCTIONS-----------------#
#----------------------------------------#
#
#Function to check if a file is an image
#
def is_image(file_path):
    try:
        Image.open(file_path).verify()
        return True
    except (IOError, SyntaxError):
        return False

########################
#Get path to rembg model
def get_model_path():
    if use_custom_model.get():
        if custom_model_path:
            return custom_model_path
        else:
            messagebox.showwarning("Model Selection", "No custom model selected. Using default.")
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.dirname(__file__)
    return os.path.join(base_path, "u2net.onnx")

def get_ffmpeg_path():
    # Use system-installed ffmpeg if available
    if shutil.which("ffmpeg"):
        return "ffmpeg"

    # Fallback to bundled ffmpeg for PyInstaller
    if hasattr(sys, "_MEIPASS"):
        fallback_path = os.path.join(sys._MEIPASS, "ffmpeg.exe")
        if os.path.exists(fallback_path):
            return fallback_path

    # Absolute fallback (only useful if debugging)
    return "ffmpeg"

######################
#Toggle model selector
def toggle_custom_model():
    global custom_model_path
    if use_custom_model.get():
        path = filedialog.askopenfilename(
            title="Select Custom ONNX Model",
            filetypes=[("ONNX Model Files", "*.onnx")]
        )
        if path:
            custom_model_path = path
            model_label["text"] = f"Using: {os.path.basename(path)}"
        else:
            use_custom_model.set(0)
            custom_model_path = None
            model_label["text"] = "Using bundled model"
    else:
        custom_model_path = None
        model_label["text"] = "Using bundled model"

###########################
#Generate masks using rembg

def generate_masks(folder_path, image_files, show_popup=True):
    global current_progress
    model_path = get_model_path()
    print(f"Using model path: {model_path}")
    session = new_session(model_name="u2net", model_path=model_path)
    folder = Path(folder_path)

    for file in image_files:
        if file.is_file() and is_image(file):
            input_path = str(file)
            output_path = str(file.parent / (file.stem + file.suffix + ".mask.png"))
            print(f"Processing file: {input_path}")

            try:
                with open(input_path, 'rb') as i:
                    input_data = i.read()
                    mask_data = remove(input_data, session=session, only_mask=True)

                mask_image = Image.open(io.BytesIO(mask_data)).convert('L')
                mask_image = mask_image.point(lambda p: p > 128 and 255)
                mask_image = mask_image.convert('1')

                mask_image.save(output_path, format='PNG')
                print(f"Mask saved to: {output_path}")
            except Exception as e:
                print(f"Error processing {file}: {e}")

        current_progress += 1
        progress_bar["value"] = current_progress
        root.update_idletasks()

    if bake_postshot.get():
        bake_masks_to_alpha(folder_path, image_files, show_popup=show_popup)
    elif show_popup:
        messagebox.showinfo("EIPC Mask Generator", "Mask generation complete!")

#######################################
#Bake image + mask into transparent PNG
def bake_masks_to_alpha(folder_path, image_files, show_popup=True):
    global current_progress
    folder = Path(folder_path)
    parent = folder.parent
    postshot_folder = parent / f"{folder.name}_Postshot"
    postshot_folder.mkdir(exist_ok=True)

    for image_path in image_files:
        mask_path = image_path.with_name(image_path.name + ".mask.png")
        if not mask_path.exists():
            print(f"No mask for {image_path.name}, skipping.")
            continue

        try:
            # Use PIL to correct EXIF orientation
            pil_img = Image.open(image_path)
            pil_img = ImageOps.exif_transpose(pil_img).convert("RGBA")
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGBA2BGRA)

            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if img is None or mask is None:
                print(f"Error reading {image_path} or its mask.")
                continue

            if img.shape[:2] != mask.shape[:2]:
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

            img[:, :, 3] = mask
            out_path = postshot_folder / (image_path.stem + ".png")
            cv2.imwrite(str(out_path), img)
            print(f"Baked: {out_path}")

        except Exception as e:
            print(f"Error processing {image_path}: {e}")

        current_progress += 1
        progress_bar["value"] = current_progress
        root.update_idletasks()

    if show_popup:
        messagebox.showinfo("EIPC Mask Generator", f"Baked images saved to: {postshot_folder}")


###############
#GENERATE MASKS 
def choose_folder_generate():
    global selected_generate_folder
    selected_generate_folder = filedialog.askdirectory()
    if selected_generate_folder:
        progress_label["text"] = f"Selected: {selected_generate_folder}"

def compute_masks():
    global total_steps, current_progress
    if selected_generate_folder:
        folder = Path(selected_generate_folder)
        image_files = [
            f for f in folder.rglob('*.*')
            if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.psd']
        ]
        count = len(image_files)
        total_steps = count + (count if bake_postshot.get() else 0)
        current_progress = 0
        progress_bar["value"] = 0
        progress_bar["maximum"] = total_steps
        progress_label["text"] = "Processing..."
        generate_masks(selected_generate_folder, image_files)
        progress_label["text"] = "Completed!"
    else:
        messagebox.showwarning("No folder selected", "Please choose a folder first.")

##############
#PNG COMPOSITE
def choose_folder_bake_only():
    global selected_combine_folder
    selected_combine_folder = filedialog.askdirectory()
    if selected_combine_folder:
        progress_label["text"] = f"Selected: {selected_combine_folder}"

def compute_bake_only():
    global total_steps, current_progress
    if selected_combine_folder:
        folder = Path(selected_combine_folder)
        image_files = [
            f for f in folder.iterdir()
            if (
                f.is_file() and
                f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.psd'] and
                not f.name.endswith(".mask.png")
            )
        ]
        total_steps = len(image_files)
        current_progress = 0
        progress_bar["value"] = 0
        progress_bar["maximum"] = total_steps
        progress_label["text"] = "Processing..."
        bake_masks_to_alpha(selected_combine_folder, image_files)
        progress_label["text"] = "Completed!"
    else:
        messagebox.showwarning("No folder selected", "Please choose a folder first.")

######################
#VIDEO SPLIT TO FRAMES
def choose_folder_videosToSplit():
    global selected_videosToSplit
    selected_videosToSplit = filedialog.askdirectory()
    if selected_videosToSplit:
        progress_label["text"] = f"Selected: {selected_videosToSplit}"

def choose_frame_save_folder():
    global selected_frame_save_folder
    selected_frame_save_folder = filedialog.askdirectory()
    if selected_frame_save_folder:
        progress_label["text"] = f"Selected: {selected_frame_save_folder}"

def compute_split_frames():
    if not selected_videosToSplit or not selected_frame_save_folder:
        messagebox.showwarning("Missing Folder", "Please select both video and save folders.")
        return

    video_folder = Path(selected_videosToSplit)
    supported_extensions = [
    "*.mp4", "*.mov", "*.mkv", "*.webm", "*.avi",
    "*.wmv", "*.flv", "*.m4v", "*.mpg", "*.mpeg",
    "*.ts", "*.3gp", "*.ogv"
    ]
    video_files = [f for ext in supported_extensions for f in video_folder.glob(ext)] #video filetype support
    if not video_files:
        messagebox.showwarning("No Videos Found", "No videos found in the selected folder.")
        return

    mode = frame_mode.get()
    try:
        if mode == "fps":
            fps = float(fps_entry.get())
            process_videos(video_files, selected_frame_save_folder, fps=fps)
        elif mode == "total":
            total = int(total_frames_entry.get())
            process_videos(video_files, selected_frame_save_folder, total_frames=total)
        else:
            raise ValueError("Invalid mode")
    except Exception as e:
        messagebox.showerror("Invalid Input", str(e))


def extract_frames_from_video(video_path, output_folder, fps=None, total_frames=None):
    try:
        os.makedirs(output_folder, exist_ok=True)

        video_capture = cv2.VideoCapture(video_path)
        frame_total = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_indices = []
        if fps is not None:
            video_fps = video_capture.get(cv2.CAP_PROP_FPS)
            frame_interval = int(video_fps / fps)
            frame_indices = list(range(0, frame_total, frame_interval))
        elif total_frames is not None:
            frame_indices = np.linspace(0, frame_total - 1, total_frames, dtype=int).tolist()
        frame_idx_set = set(frame_indices)
        current_frame = 0
        saved_frame_count = 0

        video_name = os.path.splitext(os.path.basename(video_path))[0]

        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            if current_frame in frame_idx_set:
                filename = os.path.join(output_folder, f"{video_name}_{saved_frame_count:05d}.png")
                cv2.imwrite(filename, frame)
                saved_frame_count += 1

            current_frame += 1

        video_capture.release()

    except Exception as e:
        print(f"Error processing video {video_path}: {e}")

def extract_frames_ffmpeg(video_path, output_dir, fps=None, total_frames=None):
    os.makedirs(output_dir, exist_ok=True)
    output_pattern = os.path.join(output_dir, os.path.splitext(os.path.basename(video_path))[0] + "_%05d.png")

    cmd = [get_ffmpeg_path(), "-i", str(video_path)]

    if fps:
        # Use the requested FPS
        cmd += ["-vf", f"fps={fps}"]
    elif total_frames:
        # Estimate fps needed to produce approx total_frames over video duration
        # Get video duration in seconds
        result = subprocess.run(
            [get_ffmpeg_path(), "-i", str(video_path)],
            stderr=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            text=True
        )
        duration = None
        for line in result.stderr.splitlines():
            if "Duration" in line:
                duration_str = line.split("Duration:")[1].split(",")[0].strip()
                h, m, s = map(float, duration_str.replace(":", " ").split())
                duration = h * 3600 + m * 60 + s
                break

        if duration and duration > 0:
            est_fps = total_frames / duration
            cmd += ["-vf", f"fps={est_fps:.3f}"]
        else:
            # fallback fps
            cmd += ["-vf", "fps=1"]
    else:
        raise ValueError("Must specify either fps or total_frames")

    cmd += ["-vsync", "vfr", "-q:v", "2", output_pattern]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def process_videos(video_paths, output_root_folder, fps=None, total_frames=None):
    for video_path in video_paths:
        video_title = os.path.splitext(os.path.basename(video_path))[0]
        output_folder = os.path.join(output_root_folder, video_title)

        print(f"Processing {video_path}...")

        if use_ffmpeg.get():
            extract_frames_ffmpeg(video_path, output_folder, fps=fps, total_frames=total_frames)
        else:
            extract_frames_from_video(video_path, output_folder, fps=fps, total_frames=total_frames)

        print(f"Frames saved to: {output_folder}")
        
        # Optional mask + composite generation
        if auto_generate_masks.get():
            image_files = [
                f for f in Path(output_folder).iterdir()
                if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']
            ]
            generate_masks(output_folder, image_files, show_popup=False)
        
        if auto_generate_composites.get():
            image_files = [
                f for f in Path(output_folder).iterdir()
                if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png'] and not f.name.endswith(".mask.png")
            ]
            bake_masks_to_alpha(output_folder, image_files, show_popup=False)

#----------------------------------------#
#-------------- Set up GUI --------------#
#----------------------------------------#
root = Tk()
root.title("EIPC Mask Generator v1.3")

notebook = ttk.Notebook(root)
tab_generate = Frame(notebook)
tab_combine = Frame(notebook)
tab_split = Frame(notebook)

notebook.add(tab_generate, text="Generate Masks")
notebook.add(tab_combine, text="Combine Pre-existing Masks")
notebook.add(tab_split, text="Video to Frames")
notebook.pack(expand=True, fill="both")


#button styles here

########################
# Tab 1 — Generate Masks
Label(tab_generate, text="Select the folder containing images").pack(pady=5)
Button(tab_generate, text="Choose Image Folder", command=choose_folder_generate).pack(pady=5)

use_custom_model = IntVar()
Checkbutton(tab_generate, text="Use Custom Model", variable=use_custom_model, command=toggle_custom_model).pack(pady=5)
model_label = Label(tab_generate, text="Using bundled model")
model_label.pack(pady=5)

bake_postshot = IntVar()
Checkbutton(tab_generate, text="Auto-generate Postshot PNGs", variable=bake_postshot).pack(pady=5)

Button(tab_generate, text="Compute", command=compute_masks).pack(pady=5)

################################
# Tab 2 — Combine Existing Masks
Label(tab_combine, text="Select Folder of Images + Masks:").pack(pady=5)
Button(tab_combine, text="Choose Image Folder", command=choose_folder_bake_only).pack(pady=10)

Label(tab_combine, text="Generate Postshot PNGs").pack(pady=5)
Button(tab_combine, text="Compute", command=compute_bake_only).pack(pady=5)

#########################
# Tab 3 - Video to Frames
Label(tab_split, text="Step 1:").pack(pady=0)
Button(tab_split, text="Choose Video Folder", command=choose_folder_videosToSplit).pack(pady=0)

Label(tab_split, text="Step 2:").pack(pady=0)
Button(tab_split, text="Choose Frame Save Folder", command=choose_frame_save_folder).pack(pady=0)

frame_mode = StringVar(value='fps') #default
Label(tab_split, text=" Step 3: Extraction Mode:").pack(pady=0)

frame_mode_fps = Radiobutton(tab_split, text="Frames Per Second", variable=frame_mode, value="fps")
frame_mode_fps.pack()
fps_entry = Entry(tab_split)
fps_entry.pack()

frame_mode_total = Radiobutton(tab_split, text="Total # Frames", variable=frame_mode, value="total")
frame_mode_total.pack()
total_frames_entry = Entry(tab_split)
total_frames_entry.pack(pady=0)

Label(tab_split, text="Step 4: Choose Processing Settings").pack(pady=0)
#######
#ffmpeg
use_ffmpeg = IntVar()
Checkbutton(tab_split, text="Use FFmpeg (RECOMMENDED)", variable=use_ffmpeg).pack(pady=0)

auto_generate_masks = IntVar()
Checkbutton(tab_split, text="Auto-generate Masks after Extract", variable=auto_generate_masks).pack(pady=0)
auto_generate_composites = IntVar()
Checkbutton(tab_split, text="Auto-generate Postshot PNGs", variable=auto_generate_composites).pack(pady=0)

Label(tab_split, text="Step 5:").pack(pady=0)
Button(tab_split, text="Compute", command=compute_split_frames).pack(pady=0)

####################
# Shared Progress UI
progress_label = Label(root, text="")
progress_label.pack(pady=5)

progress_bar = ttk.Progressbar(root, orient="horizontal", length=300, mode="determinate")
progress_bar.pack(pady=10)

#----------------------------------------#
#------------------MAIN------------------#
#----------------------------------------#
root.mainloop()
