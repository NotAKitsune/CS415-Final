import tensorflow as tf
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def load_and_preprocess_image(image_path):
    """Load and preprocess a single image."""
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)

def test_models():
    # Path to the new test data
    test_dir = 'New_Tests'
    
    # Get list of models
    model_files = [f for f in os.listdir() if f.startswith('wildlife_classifier_') and f.endswith('.h5')]
    
    if not model_files:
        raise ValueError("No trained models found!")
    
    # Get animal classes from test directory
    animal_classes = sorted([d for d in os.listdir(test_dir) 
                           if os.path.isdir(os.path.join(test_dir, d))])
    
    # Initialize results dictionary
    results = {
        'per_model': {},
        'per_animal': {},
        'overall': {}
    }
    
    # Test each model
    for model_file in model_files:
        model_name = model_file.replace('wildlife_classifier_', '').replace('.h5', '')
        print(f"\nTesting {model_name}...")
        
        model = load_model(model_file)
        model_results = {}
        
        for animal in animal_classes:
            animal_dir = os.path.join(test_dir, animal)
            correct = 0
            total = 0
            
            # Process each image in the animal directory
            for img_file in os.listdir(animal_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(animal_dir, img_file)
                    img = load_and_preprocess_image(img_path)
                    
                    # Get prediction
                    prediction = model.predict(img, verbose=0)
                    predicted_class = np.argmax(prediction[0])
                    
                    # Update counts
                    total += 1
                    if predicted_class == animal_classes.index(animal):
                        correct += 1
            
            accuracy = correct / total if total > 0 else 0
            model_results[animal] = accuracy
            
            # Update per-animal results
            if animal not in results['per_animal']:
                results['per_animal'][animal] = {}
            results['per_animal'][animal][model_name] = accuracy
        
        # Calculate overall accuracy for this model
        overall_accuracy = sum(model_results.values()) / len(model_results)
        results['overall'][model_name] = overall_accuracy
        results['per_model'][model_name] = model_results
    
    return results, animal_classes

def create_visualizations(results, animal_classes):
    # 1. Per-model accuracy for each animal
    for model_name, animal_results in results['per_model'].items():
        plt.figure(figsize=(12, 6))
        animals = list(animal_results.keys())
        accuracies = list(animal_results.values())
        
        plt.bar(animals, accuracies)
        plt.title(f'Accuracy by Animal Class - {model_name}')
        plt.xlabel('Animal Class')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'accuracy_{model_name}.png')
        plt.close()
    
    # 2. Per-animal comparison across models
    plt.figure(figsize=(12, 6))
    model_names = list(results['per_model'].keys())
    x = np.arange(len(animal_classes))
    width = 0.8 / len(model_names)
    
    for i, model_name in enumerate(model_names):
        accuracies = [results['per_model'][model_name][animal] for animal in animal_classes]
        plt.bar(x + i * width, accuracies, width, label=model_name)
    
    plt.xlabel('Animal Class')
    plt.ylabel('Accuracy')
    plt.title('Model Comparison by Animal Class')
    plt.xticks(x + width * (len(model_names) - 1) / 2, animal_classes, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('model_comparison_by_animal.png')
    plt.close()
    
    # 3. Overall model comparison
    plt.figure(figsize=(10, 6))
    models = list(results['overall'].keys())
    accuracies = list(results['overall'].values())
    
    plt.bar(models, accuracies)
    plt.title('Overall Model Accuracy Comparison')
    plt.xlabel('Model')
    plt.ylabel('Average Accuracy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('overall_model_comparison.png')
    plt.close()
    
    # 4. Heatmap of all results
    plt.figure(figsize=(12, 8))
    heatmap_data = np.zeros((len(animal_classes), len(model_names)))
    
    for i, animal in enumerate(animal_classes):
        for j, model in enumerate(model_names):
            heatmap_data[i, j] = results['per_model'][model][animal]
    
    sns.heatmap(heatmap_data, 
                xticklabels=model_names, 
                yticklabels=animal_classes, 
                annot=True, 
                fmt='.2f', 
                cmap='YlOrRd')
    
    plt.title('Model Performance Heatmap')
    plt.tight_layout()
    plt.savefig('performance_heatmap.png')
    plt.close()
    
    # Save numerical results
    with open('test_results.json', 'w') as f:
        json.dump(results, f, indent=4)

def main():
    try:
        print("Starting model testing...")
        results, animal_classes = test_models()
        
        print("\nGenerating visualizations...")
        create_visualizations(results, animal_classes)
        
        print("\nTesting completed successfully!")
        print("Results have been saved to 'test_results.json'")
        print("Visualizations have been saved as PNG files")
        
        # Print summary of results
        print("\nOverall Model Performance:")
        for model, accuracy in results['overall'].items():
            print(f"{model}: {accuracy:.2%}")
            
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise e

if __name__ == "__main__":
    main()