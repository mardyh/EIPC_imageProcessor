import os
import sys
from pathlib import Path
from tkinter import Tk, Button, Label, Checkbutton, IntVar, filedialog, messagebox, ttk, Frame
from rembg import remove, new_session
from PIL import Image, ImageOps
import io


#--------------VARS-----------------#
custom_model_path = None  # Will store user-selected model path if applicable
selected_generate_folder = None
selected_combine_folder = None

total_steps = 0
current_progress = 0


#--------------FUNCTIONS-----------------#

# Function to check if a file is an image
def is_image(file_path):
    try:
        Image.open(file_path).verify()
        return True
    except (IOError, SyntaxError):
        return False

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

# Generate masks using rembg
def generate_masks(folder_path):
    model_path = get_model_path()
    print(f"Using model path: {model_path}")
    session = new_session(model_name="u2net", model_path=model_path)
    folder = Path(folder_path)
    image_files = [
        f for f in folder.rglob('*.*')
        if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.psd']
    ]
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

    # Automatically run bake if checkbox is checked
    if bake_postshot.get():
        bake_masks_to_alpha(folder_path)
    else:
        messagebox.showinfo("EIPC Mask Generator", "Mask generation complete!")


# Bake image + mask into transparent PNG
def bake_masks_to_alpha(folder_path, image_files):
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
            img = Image.open(image_path)
            img = ImageOps.exif_transpose(img).convert("RGBA")

            mask = Image.open(mask_path).convert("L")

            if mask.size != img.size:
                mask = mask.resize(img.size, Image.Resampling.NEAREST)

            img.putalpha(mask)
            out_path = postshot_folder / (image_path.stem + ".png")
            img.save(out_path, format="PNG", optimize=True)

            print(f"Baked: {out_path}")
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

        current_progress += 1
        progress_bar["value"] = current_progress
        root.update_idletasks()

    messagebox.showinfo("EIPC Mask Generator", f"Baked images saved to: {postshot_folder}")


# Choose folder for generating masks
def choose_folder_generate():
    global selected_generate_folder
    selected_generate_folder = filedialog.askdirectory()
    if selected_generate_folder:
        progress_label["text"] = f"Selected: {selected_generate_folder}"

def compute_masks():
    if selected_generate_folder:
        progress_bar["value"] = 0
        progress_label["text"] = "Processing..."
        generate_masks(selected_generate_folder)
        progress_label["text"] = "Completed!"
    else:
        messagebox.showwarning("No folder selected", "Please choose a folder first.")

# Choose folder for baking only (no mask generation)
def choose_folder_bake_only():
    global selected_combine_folder
    selected_combine_folder = filedialog.askdirectory()
    if selected_combine_folder:
        progress_label["text"] = f"Selected: {selected_combine_folder}"
    
    if folder_path:
        progress_bar["value"] = 0
        progress_label["text"] = "Processing..."
        bake_masks_to_alpha(folder_path)
        progress_label["text"] = "Completed!"

def compute_bake_only():
    global total_steps, current_progress
    if selected_combine_folder:
        folder = Path(selected_combine_folder)
        image_files = [
            f for f in folder.iterdir()
            if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.psd']
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



# Toggle model selector
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

# Set up GUI
root = Tk()
root.title("EIPC Mask Generator v1.2")

notebook = ttk.Notebook(root)
tab_generate = Frame(notebook)
tab_combine = Frame(notebook)
notebook.add(tab_generate, text="Generate Masks")
notebook.add(tab_combine, text="Combine Pre-existing Masks")
notebook.pack(expand=True, fill="both")

# Tab 1 — Generate Masks
Label(tab_generate, text="Select the folder containing images:").pack(pady=10)
Button(tab_generate, text="Choose Image Folder", command=choose_folder_generate).pack(pady=5)

use_custom_model = IntVar()
Checkbutton(tab_generate, text="Use Custom Model", variable=use_custom_model, command=toggle_custom_model).pack(pady=5)
model_label = Label(tab_generate, text="Using bundled model")
model_label.pack(pady=5)

bake_postshot = IntVar()
Checkbutton(tab_generate, text="Generate Postshot-Compatible Images", variable=bake_postshot).pack(pady=5)

Button(tab_generate, text="Compute Masks", command=compute_masks).pack(pady=10)


# Tab 2 — Combine Existing Masks
Label(tab_combine, text="Select folder of images + masks:").pack(pady=10)
Button(tab_combine, text="Choose Image Folder", command=choose_folder_bake_only).pack(pady=5)

Button(tab_combine, text="Combine Masks into Alpha", command=compute_bake_only).pack(pady=5)

# Shared Progress UI
progress_label = Label(root, text="")
progress_label.pack(pady=5)

progress_bar = ttk.Progressbar(root, orient="horizontal", length=300, mode="determinate")
progress_bar.pack(pady=10)

root.mainloop()
