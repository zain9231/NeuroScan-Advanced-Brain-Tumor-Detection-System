import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import cv2
from PIL import Image, ImageTk
import numpy as np
import threading
import itertools
from preprocessing import preprocess_image
from augmentation import augment_dataset
from model import TumorDetector

class CreateToolTip:
    """A class to create tooltips for widgets."""

    def __init__(self, widget, text='widget info'):
        self.widget = widget
        self.text = text
        self.tipwindow = None
        self.id = None
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.leave)

    def enter(self, event=None):
        self.schedule()

    def leave(self, event=None):
        self.unschedule()
        self.hidetip()

    def schedule(self):
        self.unschedule()
        self.id = self.widget.after(500, self.showtip)

    def unschedule(self):
        id = self.id
        self.id = None
        if id:
            self.widget.after_cancel(id)

    def showtip(self):
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                         background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                         font=("tahoma", "8", "normal"))
        label.pack(ipadx=1)

    def hidetip(self):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()

class ImageDetectionWindow:
    def __init__(self, master):
        self.master = master
        self.window = tk.Toplevel(self.master)
        self.window.title("NeuroScan Tumor Detection")
        self.window.geometry("1000x800")  # Increased initial window size

        self.tumor_detector = TumorDetector()

        self.current_image = None
        self.original_image = None
        self.processed_image = None
        self.detections = None
        self.zoom_factor = 1.0

        self.create_widgets()

    def create_widgets(self):
        self.create_title()
        self.create_menu()
        
        # Main content frame
        main_frame = tk.Frame(self.window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left frame for image and controls
        left_frame = tk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.create_top_frame(left_frame)
        self.create_image_frame(left_frame)
        self.create_control_frame(left_frame)
        
        # Right frame for info
        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.create_info_frame(right_frame)
        
        self.create_bottom_strip()
        self.clear_info_text()

    def create_title(self):
        title_frame = tk.Frame(self.window, bg="#4a7abc")
        title_frame.pack(side=tk.TOP, fill=tk.X)
        title_label = tk.Label(title_frame, text="NeuroScan Real-Time Tumor Detection", 
                               font=("Helvetica", 16, "bold"), fg="white", bg="#4a7abc", pady=10)
        title_label.pack()
        ttk.Separator(self.window, orient='horizontal').pack(fill='x')

    def create_menu(self):
        menubar = tk.Menu(self.window)
        self.window.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Image", command=self.select_image)
        file_menu.add_command(label="Save Results", command=self.save_results)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.close)

        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="User Guide", command=self.show_user_guide)
        help_menu.add_command(label="About", command=self.show_about)

    def create_top_frame(self, parent):
        self.top_frame = tk.Frame(parent)
        self.top_frame.pack(fill=tk.X, pady=10)

        self.select_button = tk.Button(self.top_frame, text="Select Image", command=self.select_image)
        self.select_button.pack(side=tk.LEFT, padx=(0, 10))

        self.reset_button = tk.Button(self.top_frame, text="Reset", command=self.reset, state=tk.DISABLED)
        self.reset_button.pack(side=tk.LEFT)

        self.batch_process_button = tk.Button(self.top_frame, text="Batch Process", command=self.batch_process)
        self.batch_process_button.pack(side=tk.LEFT, padx=(10, 0))

    def create_image_frame(self, parent):
        self.image_frame = tk.Frame(parent)
        self.image_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.image_label = tk.Label(self.image_frame)
        self.image_label.pack(fill=tk.BOTH, expand=True)

    def create_control_frame(self, parent):
        control_frame = tk.Frame(parent)
        control_frame.pack(fill=tk.X, pady=10)

        self.toggle_button = tk.Button(control_frame, text="Show Original", command=self.toggle_image, state=tk.DISABLED)
        self.toggle_button.pack(side=tk.LEFT, padx=(0, 10))

        self.zoom_in_button = tk.Button(control_frame, text="Zoom In", command=self.zoom_in)
        self.zoom_in_button.pack(side=tk.LEFT, padx=(0, 5))

        self.zoom_out_button = tk.Button(control_frame, text="Zoom Out", command=self.zoom_out)
        self.zoom_out_button.pack(side=tk.LEFT)

        self.confidence_label = tk.Label(control_frame, text="Confidence Threshold:")
        self.confidence_label.pack(side=tk.LEFT, padx=(10, 5))

        self.confidence_slider = tk.Scale(control_frame, from_=0, to=1, resolution=0.1, orient=tk.HORIZONTAL, command=self.update_confidence)
        self.confidence_slider.set(0.5)
        self.confidence_slider.pack(side=tk.LEFT, expand=True, fill=tk.X)

    def create_info_frame(self, parent):
        self.info_frame = tk.Frame(parent)
        self.info_frame.pack(fill=tk.BOTH, expand=True, padx=(10, 0))

        self.result_label = tk.Label(self.info_frame, text="", font=("TkDefaultFont", 12, "bold"))
        self.result_label.pack(pady=5)

        self.info_text = tk.Text(self.info_frame, wrap=tk.WORD, state=tk.DISABLED)
        self.info_text.pack(fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(self.info_frame, command=self.info_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.info_text.config(yscrollcommand=scrollbar.set)

    def create_bottom_strip(self):
        bottom_frame = tk.Frame(self.window, bg="#4a7abc", height=25)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)
        bottom_frame.pack_propagate(False)

    def select_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if file_path:
            self.process_image(file_path)

    def process_image(self, file_path):
        try:
            image = cv2.imread(file_path)
            if image is None:
                raise ValueError("Failed to read the image file.")
            self.original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            self.detections = self.tumor_detector.detect_tumor(self.original_image, self.confidence_slider.get())
            self.processed_image = self.tumor_detector.draw_detections(self.original_image.copy(), self.detections)
            self.display_image(self.processed_image)
            self.current_image = 'processed'

            if len(self.detections) > 0:
                self.result_label.config(text="Tumor Detected", fg="red")
                self.update_info_text(self.detections)
                self.toggle_button.config(state=tk.NORMAL)
            else:
                self.result_label.config(text="No Tumor Detected", fg="green")
                self.clear_info_text(no_tumor=True)
                self.toggle_button.config(state=tk.DISABLED)

            self.reset_button.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while processing the image: {str(e)}")
            self.reset()

    def display_image(self, image):
        if image is None:
            return
        
        h, w = image.shape[:2]
        aspect_ratio = w / h

        # Calculate new dimensions while maintaining aspect ratio
        new_w = int(640 * self.zoom_factor)
        new_h = int(new_w / aspect_ratio)

        if new_h > 480:
            new_h = 480
            new_w = int(new_h * aspect_ratio)

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create a black background image
        background = Image.new('RGB', (640, 480), (0, 0, 0))
        
        # Convert OpenCV image to PIL Image
        pil_image = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        
        # Calculate position to paste the resized image
        x = (640 - new_w) // 2
        y = (480 - new_h) // 2
        
        # Paste the resized image onto the background
        background.paste(pil_image, (x, y))
        
        imgtk = ImageTk.PhotoImage(image=background)
        self.image_label.imgtk = imgtk
        self.image_label.configure(image=imgtk)

    def update_info_text(self, detections):
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete('1.0', tk.END)
        
        for i, det in enumerate(detections, 1):
            detection = {
                'detection_number': i,
                'class': int(det[5]),
                'confidence': det[4],
                'bbox': det[:4]
            }
            formatted_result = self.tumor_detector.format_detection_result(detection, self.original_image.shape)
            self.info_text.insert(tk.END, formatted_result)
        
        self.info_text.config(state=tk.DISABLED)

    def clear_info_text(self, no_tumor=False):
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete('1.0', tk.END)
        if no_tumor:
            self.info_text.insert(tk.END, "No tumors detected in this scan.")
        else:
            self.info_text.insert(tk.END, "No image loaded. Please select an image for tumor detection.")
        self.info_text.config(state=tk.DISABLED)

    def reset(self):
        self.image_label.config(image='')
        self.result_label.config(text="")
        self.clear_info_text()
        self.reset_button.config(state=tk.DISABLED)
        self.toggle_button.config(state=tk.DISABLED)
        self.original_image = None
        self.processed_image = None
        self.detections = None
        self.current_image = None
        self.zoom_factor = 1.0
        self.confidence_slider.set(0.5)

    def zoom_in(self):
        if self.current_image is not None:
            self.zoom_factor = min(self.zoom_factor * 1.2, 3.0)
            self.update_image()

    def zoom_out(self):
        if self.current_image is not None:
            self.zoom_factor = max(self.zoom_factor / 1.2, 0.5)
            self.update_image()

    def update_image(self):
        if self.current_image == 'processed':
            self.display_image(self.processed_image)
        else:
            self.display_image(self.original_image)

    def toggle_image(self):
        if self.current_image == 'processed':
            self.current_image = 'original'
            self.display_image(self.original_image)
            self.toggle_button.config(text="Show Processed")
        else:
            self.current_image = 'processed'
            self.display_image(self.processed_image)
            self.toggle_button.config(text="Show Original")

    def update_confidence(self, value):
        if self.original_image is not None:
            self.detections = self.tumor_detector.detect_tumor(self.original_image, float(value))
            self.processed_image = self.tumor_detector.draw_detections(self.original_image.copy(), self.detections)
            if self.current_image == 'processed':
                self.display_image(self.processed_image)
            self.update_info_text(self.detections)

    def batch_process(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            try:
                output_folder = os.path.join(folder_path, "processed")
                os.makedirs(output_folder, exist_ok=True)
                
                image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
                total_files = len(image_files)
                
                progress_window = tk.Toplevel(self.window)
                progress_window.title("Batch Processing")
                progress_bar = ttk.Progressbar(progress_window, length=300, mode='determinate')
                progress_bar.pack(padx=10, pady=10)
                progress_label = tk.Label(progress_window, text="Processing images...")
                progress_label.pack(pady=5)
                
                for i, image_file in enumerate(image_files):
                    input_path = os.path.join(folder_path, image_file)
                    output_path = os.path.join(output_folder, f"processed_{image_file}")
                    
                    image = cv2.imread(input_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    detections = self.tumor_detector.detect_tumor(image, self.confidence_slider.get())
                    processed_image = self.tumor_detector.draw_detections(image.copy(), detections)
                    
                    cv2.imwrite(output_path, cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))
                    
                    progress = (i + 1) / total_files * 100
                    progress_bar['value'] = progress
                    progress_label.config(text=f"Processing image {i+1} of {total_files}")
                    progress_window.update()
                
                progress_window.destroy()
                messagebox.showinfo("Batch Processing Complete", f"Processed {total_files} images. Results saved in {output_folder}")
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred during batch processing: {str(e)}")

    def save_results(self):
        if self.processed_image is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
            if file_path:
                cv2.imwrite(file_path, cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2BGR))
                messagebox.showinfo("Save Successful", f"Results saved to {file_path}")

    def show_user_guide(self):
        """Show the user guide dialog without sound."""
        dialog = tk.Toplevel(self.window)
        dialog.title("NeuroScan User Guide")
        self.center_window(dialog, 400, 450)  # Adjusted size to accommodate more text
        
        guide_text = """NeuroScan Tumor Detection User Guide:

         1. Select Image: Click 'Select Image' to choose a brain scan image for analysis.

         2. Batch Process: Use 'Batch Process' to analyze multiple images in a folder.

         3. Zoom: Use 'Zoom In' and 'Zoom Out' buttons to adjust image view.

         4. Confidence Threshold: Adjust the slider to set the detection confidence threshold.

         5. Toggle View: Switch between original and processed images using the 'Show Original/Processed' button.

         6. Save Results: Save the processed image with detected tumors using 'File > Save Results'.

         For more information, please refer to the documentation or contact support."""
        
        label = tk.Label(dialog, text=guide_text, wraplength=380, justify=tk.LEFT, font=("TkDefaultFont", 10))
        label.pack(pady=(20, 10), padx=10)
        
        ok_button = tk.Button(dialog, text="Close", command=dialog.destroy, font=("TkDefaultFont", 10, "bold"))
        ok_button.pack(pady=10)
        
        dialog.transient(self.window)
        dialog.grab_set()
        self.window.wait_window(dialog)


    def show_about(self):
        """Show the about dialog without sound."""
        dialog = tk.Toplevel(self.window)
        dialog.title("About NeuroScan")
        self.center_window(dialog, 350, 280)
        
        about_text = """NeuroScan Tumor Detection

        Version: 2.0
        Developed by: Zain-Ul-Abideen

        This application uses advanced image processing and machine learning 
        techniques to detect and highlight potential brain tumors in MRI scans.

        For support or more information, please contact: 
        zainulabideen2792@gmail.com"""
        
        label = tk.Label(dialog, text=about_text, wraplength=320, justify=tk.LEFT, font=("TkDefaultFont", 10))
        label.pack(pady=(20, 10))
        
        ok_button = tk.Button(dialog, text="Close", command=dialog.destroy, font=("TkDefaultFont", 10, "bold"))
        ok_button.pack(pady=10)
        
        dialog.transient(self.window)
        dialog.grab_set()
        self.window.wait_window(dialog)

    def center_window(self, window, width, height):
        """Center the window on the screen."""
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        window.geometry(f"{width}x{height}+{x}+{y}")


    def close(self):
        self.window.destroy()

class NeuroScanGUI:
    """Main class for the NeuroScan GUI application."""

    def __init__(self, master):
        self.master = master
        self.master.title("NeuroScan")
        self.center_window(self.master, 900, 720)

        self.setup_styles()
        self.create_title()

        # Initialize attributes
        self.dataset_path = tk.StringVar()
        self.num_augmented_images = tk.IntVar(value=3)
        self.progress_var = tk.DoubleVar()
        self.is_processing = False

        self.dataset_frame = tk.Frame(self.master)
    
        self.dataset_label = tk.Label(self.dataset_frame, textvariable=self.dataset_path, wraplength=880)
        self.dataset_label.pack(pady=5)
    
        self.output_note = tk.Label(self.dataset_frame, text="Output folders will be created automatically in the selected dataset folder.", 
        font=("TkDefaultFont", 9, "italic"), fg="gray50")
    
        self.create_widgets()
        self.add_tooltips()
        self.create_bottom_strip()

    def setup_styles(self):
        """Set up custom styles for widgets."""
        self.style = ttk.Style()
        self.style.layout("text.Horizontal.TProgressbar",
            [('Horizontal.Progressbar.trough',
              {'children': [('Horizontal.Progressbar.pbar',
                             {'side': 'left', 'sticky': 'ns'})],
               'sticky': 'nswe'}),
             ('Horizontal.Progressbar.label', {'sticky': ''})])
        self.style.configure("text.Horizontal.TProgressbar", text="0%")

    def create_title(self):
        """Create the title bar of the application."""
        title_frame = tk.Frame(self.master, bg="#4a7abc")
        title_frame.pack(side=tk.TOP, fill=tk.X)
        title_label = tk.Label(title_frame, text="Welcome to NeuroScan", 
                               font=("Helvetica", 16, "bold"), fg="white", bg="#4a7abc", pady=10)
        title_label.pack()
        ttk.Separator(self.master, orient='horizontal').pack(fill='x')

    def create_bottom_strip(self):
        """Create the bottom strip of the application."""
        bottom_frame = tk.Frame(self.master, bg="#4a7abc", height=25)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)
        bottom_frame.pack_propagate(False)

    def create_widgets(self):
        """Create all widgets for the application."""
        self.create_top_frame()
        self.create_main_frame()
        self.create_progress_bar()

    def create_top_frame(self):
        """Create the top frame with control buttons."""
        top_frame = tk.Frame(self.master)
        top_frame.pack(fill=tk.X, padx=10, pady=10)

        self.select_button = tk.Button(top_frame, text="Select Dataset", command=self.select_dataset)
        self.select_button.pack(side=tk.LEFT)

        self.reset_button = tk.Button(top_frame, text="Reset", command=self.reset, fg="red")
        self.reset_button.pack(side=tk.LEFT, padx=(10, 0))
        
        self.about_button = tk.Button(top_frame, text="About", command=self.show_about, fg="blue")
        self.about_button.pack(side=tk.LEFT, padx=(10, 0))

        self.exit_button = tk.Button(top_frame, text="Exit", command=self.exit_app, fg="blue")
        self.exit_button.pack(side=tk.RIGHT)

        self.detect_button = tk.Button(top_frame, text="Tumor Detection", command=self.open_detection_window)
        self.detect_button.pack(side=tk.LEFT, padx=(10, 0))

        self.dataset_frame.pack(fill=tk.X, padx=10, pady=5)

    def create_main_frame(self):
        """Create the main frame with preprocessing and augmentation options."""
        main_frame = tk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.create_preprocess_frame(main_frame)
        self.create_augment_frame(main_frame)

    def center_window(self, window, width, height):
        """Center the window on the screen."""
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        window.geometry(f"{width}x{height}+{x}+{y}")

    def open_detection_window(self):
        detection_window = ImageDetectionWindow(self.master)
        detection_window.window.protocol("WM_DELETE_WINDOW", detection_window.close)

    def show_about(self):
        """Show the about dialog."""
        about_text = """NeuroScan

        Version: 2.0

        NeuroScan is an advanced image processing tool designed for medical imaging analysis, 
        particularly focused on detecting brain tumors in neurological scans. This version 
        offers robust tumor detection capabilities along with preprocessing and augmentation 
        features for brain imaging datasets.

        Key Features:
         - Advanced brain tumor detection using deep learning algorithms
         - Dataset selection and management for individual and batch processing
         - Image preprocessing options including normalization, noise reduction, 
           skull stripping, and artifact removal
         - Data augmentation techniques such as rotation, translation, scaling, 
           flipping, elastic deformation, and more
         - Real-time tumor detection with adjustable confidence threshold
         - User-friendly interface with interactive image viewing:
           • Zoom functionality for detailed examination
           • Toggle between original and processed images
           • Detailed tumor information display
         - Batch processing capability for analyzing multiple images
         - Results saving and export functionality

        This application uses state-of-the-art machine learning techniques to detect 
        and highlight potential brain tumors in MRI scans, assisting medical professionals 
        in their diagnostic processes.

        Developed by: Zain Ul Abideen
        Under the supervision of Mr. Umair Ali

        For inquiries or support, please contact: bc200418437@vu.edu.pk

        © 2024 NeuroScan. All rights reserved.
        """

        dialog = tk.Toplevel(self.master)
        dialog.title("About NeuroScan")
        self.center_window(dialog, 790, 590)
    
        text_widget = tk.Text(dialog, wrap=tk.WORD, padx=10, pady=10)
        text_widget.pack(expand=True, fill=tk.BOTH)
        text_widget.insert(tk.END, about_text)
        text_widget.config(state=tk.DISABLED)  # Make the text read-only
    
        scrollbar = tk.Scrollbar(text_widget)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text_widget.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=text_widget.yview)
    
        ok_button = tk.Button(dialog, text="OK", command=dialog.destroy)
        ok_button.pack(pady=10)

        dialog.transient(self.master)
        dialog.grab_set()
        self.master.wait_window(dialog)

    def create_preprocess_frame(self, parent):
        """Create the preprocessing options frame."""
        self.preprocess_frame = ttk.LabelFrame(parent, text="Preprocessing")
        self.preprocess_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        ttk.Label(self.preprocess_frame, text="Preprocessing", font=("TkDefaultFont", 10, "bold")).pack(pady=5)

        self.preprocess_vars = {
            "image_normalization": tk.BooleanVar(value=True),
            "noise_reduction": tk.BooleanVar(value=True),
            "skull_stripping": tk.BooleanVar(value=True),
            "artifact_removal": tk.BooleanVar(value=True)
        }

        for key, var in self.preprocess_vars.items():
            cb = tk.Checkbutton(self.preprocess_frame, text=key.replace("_", " ").title(), variable=var)
            cb.pack(anchor="w", padx=5, pady=2)

        self.skip_preprocess_var = tk.BooleanVar(value=False)
        self.skip_preprocess_cb = tk.Checkbutton(self.preprocess_frame, text="Skip Preprocessing", 
                                                 variable=self.skip_preprocess_var,
                                                 command=self.toggle_preprocessing)
        self.skip_preprocess_cb.pack(anchor="w", padx=5, pady=2)

        self.preprocess_button = tk.Button(self.preprocess_frame, text="Apply Preprocessing", command=self.preprocess)
        self.preprocess_button.pack(pady=10)

        self.preprocess_status = tk.Label(self.preprocess_frame, text="", fg="green", font=("TkDefaultFont", 9, "bold"))
        self.preprocess_status.pack(pady=5)

    def create_augment_frame(self, parent):
        """Create the augmentation options frame."""
        self.augment_frame = ttk.LabelFrame(parent, text="Augmentation")
        self.augment_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        ttk.Label(self.augment_frame, text="Augmentation", font=("TkDefaultFont", 10, "bold")).pack(pady=5)

        self.augment_vars = {
            "rotation": tk.BooleanVar(value=True),
            "translation": tk.BooleanVar(value=True),
            "scaling": tk.BooleanVar(value=True),
            "flipping": tk.BooleanVar(value=True),
            "elastic_deformation": tk.BooleanVar(value=True),
            "intensity_adjustment": tk.BooleanVar(value=True),
            "noise_injection": tk.BooleanVar(value=True),
            "shearing": tk.BooleanVar(value=True),
            "random_cropping": tk.BooleanVar(value=True)
        }

        for key, var in self.augment_vars.items():
            cb = tk.Checkbutton(self.augment_frame, text=key.replace("_", " ").title(), variable=var)
            cb.pack(anchor="w", padx=5, pady=2)

        self.skip_augment_var = tk.BooleanVar(value=False)
        self.skip_augment_cb = tk.Checkbutton(self.augment_frame, text="Skip Augmentation", 
                                              variable=self.skip_augment_var,
                                              command=self.toggle_augmentation)
        self.skip_augment_cb.pack(anchor="w", padx=5, pady=2)

        num_images_frame = tk.Frame(self.augment_frame)
        num_images_frame.pack(fill=tk.X, padx=5, pady=5)
    
        tk.Label(num_images_frame, text="Number of augmented images:").pack(side=tk.LEFT)
    
        self.num_images_entry = tk.Entry(num_images_frame, width=5, validate="key", 
                                         validatecommand=(self.master.register(self.validate_num_images), '%P'))
        self.num_images_entry.pack(side=tk.LEFT, padx=(5, 0))
        self.num_images_entry.insert(0, str(self.num_augmented_images.get()))

        self.augment_button = tk.Button(self.augment_frame, text="Apply Augmentation", command=self.augment_data, state=tk.DISABLED)
        self.augment_button.pack(pady=10)

        self.augment_status = tk.Label(self.augment_frame, text="", fg="green", font=("TkDefaultFont", 9, "bold"))
        self.augment_status.pack(pady=5)

    def create_progress_bar(self):
        """Create the progress bar and animation label."""
        progress_frame = tk.Frame(self.master)
        progress_frame.pack(pady=(10, 2.5))

        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100, 
                                            style="text.Horizontal.TProgressbar", length=400)
        self.progress_bar.pack()
    
        self.animation_label = tk.Label(progress_frame, text="", font=("Courier", 13))
        self.animation_label.pack(pady=5)

    def add_tooltips(self):
        """Add tooltips to various widgets."""
        tooltips = {
            self.select_button: "Select a folder containing the dataset images",
            self.reset_button: "Reset all settings to default",
            self.about_button: "Show information about NeuroScan",
            self.exit_button: "Exit the application",
            self.preprocess_button: "Apply selected preprocessing techniques to the dataset",
            self.augment_button: "Apply selected augmentation techniques to the dataset",
            self.num_images_entry: "Number of augmented images to generate for each original image"
        }

        for widget, text in tooltips.items():
            CreateToolTip(widget, text)

        self.add_preprocessing_tooltips()
        self.add_augmentation_tooltips()

    def add_preprocessing_tooltips(self):
        """Add tooltips to preprocessing options."""
        tooltips = {
            "Image Normalization": "Adjust pixel intensities to a standard scale",
            "Noise Reduction": "Reduce random variations of brightness or color in images",
            "Skull Stripping": "Remove non-brain tissues from brain imaging data",
            "Artifact Removal": "Remove unwanted objects or patterns from images"
        }

        for widget in self.preprocess_frame.winfo_children():
            if isinstance(widget, tk.Checkbutton):
                text = tooltips.get(widget.cget("text"))
                if text:
                    CreateToolTip(widget, text)

    def add_augmentation_tooltips(self):
        """Add tooltips to augmentation options."""
        tooltips = {
            "Rotation": "Rotate the image by a random angle",
            "Translation": "Shift the image horizontally or vertically",
            "Scaling": "Resize the image by a random factor",
            "Flipping": "Mirror the image horizontally or vertically",
            "Elastic Deformation": "Apply random elastic distortions to the image",
            "Intensity Adjustment": "Randomly adjust brightness and contrast",
            "Noise Injection": "Add random noise to the image",
            "Shearing": "Slant the image by a random angle",
            "Random Cropping": "Extract a random portion of the image"
        }

        for widget in self.augment_frame.winfo_children():
            if isinstance(widget, tk.Checkbutton):
                text = tooltips.get(widget.cget("text"))
                if text:
                    CreateToolTip(widget, text)

    def animate_working(self, process_name):
        """Animate the working process."""
        animation = itertools.cycle(['-', '\\', '|', '/'])
        def animate():
            if self.is_processing:
                self.animation_label.config(text=f"{process_name} {next(animation)}")
                self.master.after(100, animate)
            else:
                self.animation_label.config(text="")
        self.is_processing = True
        animate()

    def select_dataset(self):
        """Open a dialog to select the dataset folder."""
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.dataset_path.set(folder_path)
            self.output_note.pack(in_=self.dataset_frame, pady=(0, 5))  
        else:
            self.dataset_path.set("")
            self.output_note.pack_forget()  
         
    def reset(self):
        """Reset all settings to default."""
        self.dataset_path.set("")
        self.output_note.pack_forget()
        self.num_augmented_images.set(3)
        for var in self.preprocess_vars.values():
            var.set(True)
        for var in self.augment_vars.values():
            var.set(True)
        self.skip_preprocess_var.set(False)
        self.skip_augment_var.set(False)
        self.toggle_preprocessing()
        self.toggle_augmentation()
        self.preprocess_status.config(text="")
        self.augment_status.config(text="")
        self.progress_var.set(0)
        self.style.configure("text.Horizontal.TProgressbar", text="0%")
        self.augment_button.config(state=tk.DISABLED)
        self.is_processing = False
        self.animation_label.config(text="")

    def toggle_preprocessing(self):
        """Toggle preprocessing options."""
        skip_preprocess = self.skip_preprocess_var.get()
        state = tk.DISABLED if skip_preprocess else tk.NORMAL
        for var in self.preprocess_vars.values():
            var.set(not skip_preprocess)
        for child in self.preprocess_frame.winfo_children():
            if isinstance(child, tk.Checkbutton) and child != self.skip_preprocess_cb:
                child.config(state=state)
        self.preprocess_button.config(state=state)

    def toggle_augmentation(self):
        """Toggle augmentation options."""
        skip_augment = self.skip_augment_var.get()
        state = tk.DISABLED if skip_augment else tk.NORMAL
        for var in self.augment_vars.values():
            var.set(not skip_augment)
        for child in self.augment_frame.winfo_children():
            if isinstance(child, tk.Checkbutton) and child != self.skip_augment_cb:
                child.config(state=state)
        self.num_images_entry.config(state=state)
        self.augment_button.config(state=state)

    def preprocess(self):
        """Start the preprocessing process."""
        if not self.dataset_path.get():
            messagebox.showerror("Error", "Please select a dataset first.")
            return
    
        self.disable_buttons()
        self.progress_var.set(0)
        self.style.configure("text.Horizontal.TProgressbar", text="0%")
    
        self.animate_working("Preprocessing")
        threading.Thread(target=self._preprocess_thread, daemon=True).start()

    def _preprocess_thread(self):
        """Thread function for preprocessing."""
        input_folder = self.dataset_path.get()
        output_folder = os.path.join(input_folder, "preprocessed")
        os.makedirs(output_folder, exist_ok=True)

        try:
            processed_images = 0
            preprocess_options = {k: v.get() for k, v in self.preprocess_vars.items()}
            total_images = len([f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            
            for i, filename in enumerate(os.listdir(input_folder)):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    input_path = os.path.join(input_folder, filename)
                    output_path = os.path.join(output_folder, f"preprocessed_{filename}")
                    
                    preprocessed_image = preprocess_image(input_path, preprocess_options)
                    cv2.imwrite(output_path, preprocessed_image)
                    processed_images += 1
                    
                    progress = (i + 1) / total_images * 100
                    self.update_progress(progress)
            
            self.master.after(0, lambda: self.show_completion_dialog(f"Preprocessing completed!\n{processed_images} images processed."))
            self.master.after(0, lambda: self.preprocess_status.config(text=f"Preprocessing completed successfully. {processed_images} images processed."))
        except Exception as e:
            self.master.after(0, lambda: messagebox.showerror("Error", f"An error occurred during preprocessing: {str(e)}"))
        finally:
            self.is_processing = False
            self.master.after(0, self.enable_buttons)
    
    def validate_num_images(self, P):
        """Validate the number of images entry."""
        if P == "":
            return True
        try:
            value = int(P)
            return value > 0
        except ValueError:
            return False

    def augment_data(self):
        """Start the augmentation process."""
        if not self.dataset_path.get():
            messagebox.showerror("Error", "Please select a dataset first.")
            return
        
        if self.skip_augment_var.get():
            messagebox.showinfo("Info", "Augmentation skipped!")
            return

        try:
            num_images = int(self.num_images_entry.get())
            if num_images <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Number of augmented images must be a positive whole number.")
            return

        self.num_augmented_images.set(num_images)  # Update the IntVar with the validated value
        self.disable_buttons()
        self.progress_var.set(0)
        self.style.configure("text.Horizontal.TProgressbar", text="0%")

        self.animate_working("Augmenting")
        threading.Thread(target=self._augment_thread, daemon=True).start()

    def _augment_thread(self):
        """Thread function for augmentation."""
        input_folder = os.path.join(self.dataset_path.get(), "preprocessed")
        if not os.path.exists(input_folder):
            input_folder = self.dataset_path.get()

        output_folder = os.path.join(self.dataset_path.get(), "augmented")

        try:
            augment_options = {k: v.get() for k, v in self.augment_vars.items()}
            processed_images = augment_dataset(input_folder, output_folder, augment_options, self.num_augmented_images.get(),
                                               progress_callback=self.update_progress)
            total_augmented = sum(os.path.isfile(os.path.join(output_folder, f)) for f in os.listdir(output_folder))
            self.master.after(0, lambda: self.show_completion_dialog(
                f"Data augmentation completed!\n"
                f"{processed_images} original images processed.\n"
                f"{total_augmented} augmented images created."
            ))
            self.master.after(0, lambda: self.augment_status.config(text=f"Augmentation completed successfully. {total_augmented} images created."))
        except Exception as e:
            error_message = f"An error occurred during data augmentation: {str(e)}"
            self.master.after(0, lambda m=error_message: messagebox.showerror("Error", m))
        finally:
            self.is_processing = False
            self.master.after(0, self.enable_buttons)

    def update_progress(self, progress):
        """Update the progress bar."""
        self.progress_var.set(progress)
        self.style.configure("text.Horizontal.TProgressbar", text=f"{progress:.1f}%")
        self.master.update_idletasks()  # Force update of the GUI

    def show_completion_dialog(self, message):
        """Show a completion dialog with the given message."""
        dialog = tk.Toplevel(self.master)
        dialog.title("Operation Completed")
        self.center_window(dialog, 300, 150) 
        
        label = tk.Label(dialog, text=message, wraplength=280)
        label.pack(pady=20)
        
        ok_button = tk.Button(dialog, text="OK", command=dialog.destroy)
        ok_button.pack(pady=10)
        
        dialog.transient(self.master)
        dialog.grab_set()
        self.master.wait_window(dialog)

    def disable_buttons(self):
        """Disable all control buttons."""
        for button in [self.select_button, self.reset_button, self.preprocess_button, self.augment_button]:
            button.config(state=tk.DISABLED)

    def enable_buttons(self):
        """Enable all control buttons."""
        for button in [self.select_button, self.reset_button, self.preprocess_button, self.augment_button]:
            button.config(state=tk.NORMAL)
    
    def exit_app(self):
        """Exit the application with a farewell message."""
        if messagebox.askyesno("Confirm Exit", "Are you sure you want to exit?"):
            dialog = tk.Toplevel(self.master)
            dialog.title("Exit Notification")
            self.center_window(dialog, 350, 340)
            
            label = tk.Label(dialog, text="Thank you for using NeuroScan Tumor Detection!\n\n"
                              "This project has been successfully completed with real-time tumor detection "
                              "and other advanced features implemented.\n\n"
                              "We hope this tool assists medical professionals in their diagnostic processes.\n\n"
                              "Developed by:\n\nZain Ul Abideen\nUnder the supervision of Mr. Umair Ali\n\n"
                              "For any inquiries or support, please contact:\nbc200418437@vu.edu.pk",
                             wraplength=320, font=("TkDefaultFont", 10))
            label.pack(pady=(20, 10))
            
            ok_button = tk.Button(dialog, text="Exit", command=self.master.destroy, font=("TkDefaultFont", 10, "bold"))
            ok_button.pack(pady=10)
            
            dialog.transient(self.master)
            dialog.grab_set()
            self.master.wait_window(dialog)

if __name__ == "__main__":
    root = tk.Tk()
    app = NeuroScanGUI(root)
    root.mainloop()