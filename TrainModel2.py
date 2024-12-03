import tensorflow as tf
import os
import time
import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

# Force CPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def process_image(file_path, label):
    """Process a single image file."""
    # Convert the path tensor to string
    file_path = tf.convert_to_tensor(file_path)
    file_path = tf.strings.as_string(file_path)
    
    # Read the image file
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = tf.cast(image, tf.float32) / 255.0
    
    return image, label

def create_dataset(base_dir, target_size=(224, 224), batch_size=32, split_ratio=(0.7, 0.15, 0.15)):
    """Creates tf.data.Dataset from directory structure"""
    print(f"Looking for classes in: {base_dir}")
    
    if not os.path.exists(base_dir):
        raise ValueError(f"Base directory {base_dir} does not exist")
    
    class_names = sorted([d for d in os.listdir(base_dir) 
                         if os.path.isdir(os.path.join(base_dir, d))])
    
    if not class_names:
        raise ValueError(f"No class directories found in {base_dir}")
    
    print(f"Found classes: {class_names}")
    
    all_paths = []
    all_labels = []
    
    print("\nScanning directories for images...")
    for class_idx, class_name in enumerate(class_names):
        class_path = os.path.join(base_dir, class_name)
        print(f"\nProcessing class {class_idx}: {class_name}")
        
        # Get all images in the class directory
        valid_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
        images = [f for f in os.listdir(class_path) 
                 if os.path.isfile(os.path.join(class_path, f)) and 
                 f.lower().endswith(valid_extensions)]
        
        print(f"Found {len(images)} images in {class_name}")
        
        for img_name in images:
            all_paths.append(os.path.join(class_path, img_name))
            all_labels.append(class_idx)
    
    # Check if we found any images
    if not all_paths:
        raise ValueError("No images found!")
    
    # Convert to tensors and shuffle
    all_paths = tf.convert_to_tensor(all_paths)
    all_labels = tf.convert_to_tensor(all_labels)
    
    # Shuffle the data
    indices = tf.range(len(all_paths))
    tf.random.shuffle(indices)
    
    all_paths = tf.gather(all_paths, indices)
    all_labels = tf.gather(all_labels, indices)
    
    # Split the data
    total_size = len(all_paths)
    train_size = int(total_size * split_ratio[0])
    val_size = int(total_size * split_ratio[1])
    
    train_paths = all_paths[:train_size]
    train_labels = all_labels[:train_size]
    
    val_paths = all_paths[train_size:train_size + val_size]
    val_labels = all_labels[train_size:train_size + val_size]
    
    test_paths = all_paths[train_size + val_size:]
    test_labels = all_labels[train_size + val_size:]
    
    # Create datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    val_dataset = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))
    
    # Configure datasets
    train_dataset = (train_dataset
        .shuffle(len(train_paths))
        .map(process_image, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE))
    
    val_dataset = (val_dataset
        .map(process_image, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE))
    
    test_dataset = (test_dataset
        .map(process_image, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE))
    
    num_classes = len(class_names)
    print(f"\nDataset prepared with {num_classes} classes")
    print(f"Training samples: {len(train_paths)}")
    print(f"Validation samples: {len(val_paths)}")
    print(f"Test samples: {len(test_paths)}")
    
    return train_dataset, val_dataset, test_dataset, num_classes

def create_control_model(input_shape=(224, 224, 3), num_classes=None):
    model = Sequential([
        Conv2D(32, 3, activation='relu', padding='same', input_shape=input_shape),
        MaxPooling2D(2),
        
        Conv2D(64, 3, activation='relu', padding='same'),
        MaxPooling2D(2),
        
        Conv2D(64, 3, activation='relu', padding='same'),
        MaxPooling2D(2),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

def create_model_1(input_shape=(224, 224, 3), num_classes=None):
    model = Sequential([
        Conv2D(32, 3, activation='relu', padding='same', input_shape=input_shape),
        Conv2D(32, 3, activation='relu', padding='same'),
        Conv2D(32, 3, activation='relu', padding='same'),
        MaxPooling2D(2),
        
        Conv2D(64, 3, activation='relu', padding='same'),
        Conv2D(64, 3, activation='relu', padding='same'),
        MaxPooling2D(2),
        
        Conv2D(128, 3, activation='relu', padding='same'),
        Conv2D(128, 3, activation='relu', padding='same'),
        MaxPooling2D(2),
        
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    return model

def create_model_2(input_shape=(224, 224, 3), num_classes=None):
    model = Sequential([
        Conv2D(64, 3, activation='relu', padding='same', input_shape=input_shape),
        MaxPooling2D(2),
        
        Conv2D(128, 3, activation='relu', padding='same'),
        MaxPooling2D(2),
        
        Conv2D(256, 3, activation='relu', padding='same'),
        MaxPooling2D(2),
        
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

def create_model_3(input_shape=(224, 224, 3), num_classes=None):
    model = Sequential([
        Conv2D(32, 2, activation='relu', padding='same', input_shape=input_shape),
        MaxPooling2D(2),
        
        Conv2D(64, 2, activation='relu', padding='same'),
        MaxPooling2D(2),
        
        Conv2D(128, 2, activation='relu', padding='same'),
        MaxPooling2D(2),
        
        Conv2D(256, 2, activation='relu', padding='same'),
        MaxPooling2D(2),
        
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])
    return model

def create_model_4(input_shape=(224, 224, 3), num_classes=None):
    model = Sequential([
        Conv2D(32, 3, activation='relu', padding='same', input_shape=input_shape),
        MaxPooling2D(2),
        
        Conv2D(64, 3, activation='relu', padding='same'),
        MaxPooling2D(2),
        
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.4),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    return model

try:
    # Load and prepare data
    data_dir = 'Data'  # Updated path to match your structure
    print(f"\nStarting data loading from: {os.path.abspath(data_dir)}")
    train_dataset, val_dataset, test_dataset, num_classes = create_dataset(data_dir)

    # Define models to train
    models = {
        'control': create_control_model,
        'model1': create_model_1,
        'model2': create_model_2,
        'model3': create_model_3,
        'model4': create_model_4
    }

    # Results storage
    results = {}

    # Train and evaluate each model
    for model_name, model_fn in models.items():
        print(f"\n=== Training {model_name} ===")
        
        # Create and compile model
        model = model_fn(num_classes=num_classes)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Measure model size
        model_size_mb = sum(tf.keras.backend.count_params(w) * 4 for w in model.weights) / (1024 * 1024)

        # Measure training time
        start_time = time.time()
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=10,
            verbose=1
        )
        training_time = time.time() - start_time

        # Measure inference time
        start_time = time.time()
        test_loss, test_accuracy = model.evaluate(test_dataset)
        inference_time = (time.time() - start_time) / len(test_dataset)

        print(f"\n{model_name} test accuracy: {test_accuracy:.4f}")

        # Save results
        results[model_name] = {
            'history': history.history,
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'training_time_hours': training_time / 3600,
            'inference_time_ms': inference_time * 1000,
            'model_size_mb': model_size_mb
        }

        # Save the model
        model.save(f'wildlife_classifier_{model_name}.h5')

    # Print and save comparative results
    print("\n=== Final Results ===")
    comparison_results = {}
    
    for model_name, result in results.items():
        print(f"{model_name}:")
        print(f"  Test Accuracy: {result['test_accuracy']:.4f}")
        print(f"  Best Validation Accuracy: {max(result['history']['val_accuracy']):.4f}")
        print(f"  Training Time: {result['training_time_hours']:.2f}h")
        print(f"  Inference Time: {result['inference_time_ms']:.2f}ms")
        print(f"  Model Size: {result['model_size_mb']:.2f}MB")
        print()
        
        comparison_results[model_name] = {
            'test_accuracy': f"{result['test_accuracy']:.4f}",
            'best_val_accuracy': f"{max(result['history']['val_accuracy']):.4f}",
            'training_time_hours': f"{result['training_time_hours']:.2f}",
            'inference_time_ms': f"{result['inference_time_ms']:.2f}",
            'model_size_mb': f"{result['model_size_mb']:.2f}"
        }

    # Save results
    with open('model_results_detailed.json', 'w') as f:
        json.dump(results, f, indent=4)

    with open('model_comparison.json', 'w') as f:
        json.dump(comparison_results, f, indent=4)

except Exception as e:
    print(f"\nError encountered: {str(e)}")
    print("\nCurrent working directory:", os.getcwd())
    raise e