import tensorflow as tf
import tkinter as tk
from tkinter import ttk, filedialog
import numpy as np
from PIL import Image, ImageTk
import os

class WildlifeClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Wildlife Classifier")
        self.root.geometry("800x600")
        
        # Store loaded models
        self.models = {}
        self.current_model = None
        self.class_names = None
        
        # Load models and class names
        self.load_models()
        
        # Create GUI
        self.create_gui()
    
    def load_models(self):
        """Load all available models from the current directory"""
        model_files = [f for f in os.listdir('.') if f.startswith('wildlife_classifier_') and f.endswith('.h5')]
        
        # Get class names from the Tests directory
        if os.path.exists('Tests'):
            self.class_names = sorted([d for d in os.listdir('Tests') 
                                     if os.path.isdir(os.path.join('Tests', d))])
        else:
            raise ValueError("Tests directory not found! Please ensure it exists in the same directory.")
        
        print("Loading models...")
        for model_file in model_files:
            model_name = model_file.replace('wildlife_classifier_', '').replace('.h5', '')
            try:
                self.models[model_name] = tf.keras.models.load_model(model_file)
                print(f"Loaded {model_name}")
            except Exception as e:
                print(f"Error loading {model_name}: {str(e)}")
    
    def create_gui(self):
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Model selection
        ttk.Label(main_frame, text="Select Model:").grid(row=0, column=0, sticky=tk.W)
        self.model_var = tk.StringVar()
        model_dropdown = ttk.Combobox(main_frame, textvariable=self.model_var)
        model_dropdown['values'] = list(self.models.keys())
        model_dropdown.grid(row=0, column=1, sticky=(tk.W, tk.E))
        model_dropdown.bind('<<ComboboxSelected>>', self.on_model_selected)
        
        # Image upload button
        ttk.Button(main_frame, text="Upload Image", command=self.upload_image).grid(row=1, column=0, columnspan=2, pady=10)
        
        # Image preview
        self.image_label = ttk.Label(main_frame)
        self.image_label.grid(row=2, column=0, columnspan=2, pady=10)
        
        # Results section
        self.results_frame = ttk.LabelFrame(main_frame, text="Classification Results", padding="10")
        self.results_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        # Progress bar
        self.progress_vars = {}
        self.progress_labels = {}
        for i, class_name in enumerate(self.class_names):
            ttk.Label(self.results_frame, text=class_name).grid(row=i, column=0, sticky=tk.W)
            
            # Progress bar for confidence
            progress_var = tk.DoubleVar()
            self.progress_vars[class_name] = progress_var
            progress_bar = ttk.Progressbar(self.results_frame, length=200, mode='determinate', variable=progress_var)
            progress_bar.grid(row=i, column=1, padx=5)
            
            # Label for percentage
            label_var = tk.StringVar()
            self.progress_labels[class_name] = label_var
            ttk.Label(self.results_frame, textvariable=label_var).grid(row=i, column=2)
    
    def on_model_selected(self, event):
        self.current_model = self.models[self.model_var.get()]
    
    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        
        if file_path and self.current_model:
            # Load and preprocess image
            image = tf.io.read_file(file_path)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, [224, 224])
            
            # Display preview
            preview = Image.open(file_path)
            preview.thumbnail((300, 300))
            photo = ImageTk.PhotoImage(preview)
            self.image_label.configure(image=photo)
            self.image_label.image = photo
            
            # Prepare image for prediction
            input_image = tf.cast(image, tf.float32) / 255.0
            input_image = tf.expand_dims(input_image, 0)
            
            # Make prediction
            predictions = self.current_model.predict(input_image)
            
            # Update progress bars and labels
            for class_name, confidence, progress_var, label_var in zip(
                self.class_names, 
                predictions[0], 
                self.progress_vars.values(), 
                self.progress_labels.values()
            ):
                confidence_pct = confidence * 100
                progress_var.set(confidence_pct)
                label_var.set(f"{confidence_pct:.1f}%")

if __name__ == "__main__":
    root = tk.Tk()
    app = WildlifeClassifierApp(root)
    root.mainloop()