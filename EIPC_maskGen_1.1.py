import os
import sys
from pathlib import Path
from tkinter import Tk, Button, Label, Checkbutton, IntVar, filedialog, messagebox, ttk
from rembg import remove, new_session
from PIL import Image
import io

custom_model_path = None  # Will store user-selected model path if applicable

# Function to check if a file is an image
def is_image(file_path):
    try:
        Image.open(file_path).verify()
        return True
    except (IOError, SyntaxError):
        return False

# Function to get the correct model path
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

# Function to generate masks
def generate_masks(folder_path):
    model_path = get_model_path()
    print(f"Using model path: {model_path}")
    session = new_session(model_name="u2net", model_path=model_path)
    folder = Path(folder_path)
    image_files = list(folder.rglob('*.*'))
    total_files = len(image_files)
    processed_files = 0

    progress_bar["maximum"] = total_files

    for file in image_files:
        if file.is_file() and is_image(file):
            input_path = str(file)
            output_path = str(file.parent / (file.stem + file.suffix + ".mask.png"))
            print(f"Processing file: {input_path}")

            try:
                with open(input_path, 'rb') as i:
                    input_data = i.read()
                    mask_data = remove(input_data, session=session, only_mask=True)

                # Convert mask to 1-bit image
                mask_image = Image.open(io.BytesIO(mask_data)).convert('L')
                mask_image = mask_image.point(lambda p: p > 128 and 255)
                mask_image = mask_image.convert('1')

                # Save the 1-bit image
                mask_image.save(output_path, format='PNG')
                print(f"Mask saved to: {output_path}")
            except Exception as e:
                print(f"Error processing {file}: {e}")

        processed_files += 1
        progress_bar["value"] = processed_files
        root.update_idletasks()

    messagebox.showinfo("EIPC Mask Generator", "Mask generation complete!")

# Function to choose folder
def choose_folder():
    folder_path = filedialog.askdirectory()
    if folder_path:
        progress_bar["value"] = 0
        progress_label["text"] = "Processing..."
        generate_masks(folder_path)
        progress_label["text"] = "Completed!"

# Function to toggle custom model selection
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

# Set up the GUI
root = Tk()
root.title("EIPC Mask Generator v1.1")

label = Label(root, text="Select the folder containing images:")
label.pack(pady=10)

button = Button(root, text="Choose Folder", command=choose_folder)
button.pack(pady=10)

# Checkbox to toggle custom model
use_custom_model = IntVar()
custom_model_checkbox = Checkbutton(root, text="Use Custom Model", variable=use_custom_model, command=toggle_custom_model)
custom_model_checkbox.pack(pady=5)

model_label = Label(root, text="Using bundled model")
model_label.pack(pady=5)

progress_label = Label(root, text="")
progress_label.pack(pady=5)

progress_bar = ttk.Progressbar(root, orient="horizontal", length=300, mode="determinate")
progress_bar.pack(pady=10)

root.mainloop()
