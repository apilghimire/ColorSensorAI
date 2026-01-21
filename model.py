import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import pickle
import json

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report)

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

import warnings
warnings.filterwarnings('ignore')

# ======= CONFIGURATION =======
CSV_FILE = 'Data.csv'  # Your merged dataset
TEST_SIZE = 0.2  # 20% for testing, 80% for training
RANDOM_STATE = 42
OUTPUT_DIR = 'Results'
MODELS_DIR = 'models'

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

class ColorClassifier:
    """Train and evaluate multiple ML models for color classification"""
    
    def __init__(self, data_file):
        self.data_file = data_file
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.models = {}
        self.results = {}
        
    def load_and_prepare_data(self):
        """Load and preprocess the dataset"""
        print("="*70)
        print("LOADING DATA")
        print("="*70)
        
        # Load CSV
        df = pd.read_csv(self.data_file)
        print(f"Total samples: {len(df)}")
        print(f"\nSamples per class:")
        print(df['Label'].value_counts())
        
        # Features and labels
        feature_columns = ['Red', 'Green', 'Blue', 'Clear', 'Distance']
        X = df[feature_columns].values
        y = df['Label'].values
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded
        )
        
        # Scale features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print(f"\nTraining samples: {len(self.X_train)}")
        print(f"Testing samples: {len(self.X_test)}")
        print("="*70 + "\n")
        
        return df
    
    def train_models(self):
        """Train all models"""
        print("="*70)
        print("TRAINING MODELS")
        print("="*70)
        
        # Define models
        self.models = {
            'SVM': SVC(kernel='rbf', random_state=RANDOM_STATE, probability=True),
            'Random_Forest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'Neural_Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, 
                                           random_state=RANDOM_STATE, early_stopping=True)
        }
        
        # Train each model
        for name, model in self.models.items():
            print(f"\nTraining {name.replace('_', ' ')}...")
            model.fit(self.X_train, self.y_train)
            
            # Predictions
            y_pred = model.predict(self.X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='weighted')
            recall = recall_score(self.y_test, y_pred, average='weighted')
            f1 = f1_score(self.y_test, y_pred, average='weighted')
            
            # Cross-validation score
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5)
            
            # Store results
            self.results[name] = {
                'model': model,
                'y_pred': y_pred,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'confusion_matrix': confusion_matrix(self.y_test, y_pred)
            }
            
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        print("\n" + "="*70 + "\n")
    
    def save_models(self):
        """Save all trained models and preprocessing objects"""
        print("="*70)
        print("SAVING MODELS")
        print("="*70)
        
        # Save each model
        for name, result in self.results.items():
            model_path = os.path.join(MODELS_DIR, f'{name}.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(result['model'], f)
            print(f"✓ Saved: {name}.pkl (Accuracy: {result['accuracy']:.4f})")
        
        # Save scaler
        scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"✓ Saved: scaler.pkl")
        
        # Save label encoder
        encoder_path = os.path.join(MODELS_DIR, 'label_encoder.pkl')
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        print(f"✓ Saved: label_encoder.pkl")
        
        # Save model metadata
        metadata = {
            'classes': self.label_encoder.classes_.tolist(),
            'feature_names': ['Red', 'Green', 'Blue', 'Clear', 'Distance'],
            'model_performance': {
                name: {
                    'accuracy': float(result['accuracy']),
                    'precision': float(result['precision']),
                    'recall': float(result['recall']),
                    'f1_score': float(result['f1_score'])
                }
                for name, result in self.results.items()
            },
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'dataset': self.data_file,
            'total_samples': len(self.X_train) + len(self.X_test)
        }
        
        metadata_path = os.path.join(MODELS_DIR, 'model_info.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"✓ Saved: model_info.json")
        
        print(f"\nAll models saved in '{MODELS_DIR}/' directory")
        print("="*70 + "\n")
    
    def save_results_csv(self):
        """Save all metrics to CSV"""
        print("Saving results to CSV...")
        
        # Model comparison
        comparison_data = []
        for name, result in self.results.items():
            comparison_data.append({
                'Model': name.replace('_', ' '),
                'Accuracy': result['accuracy'],
                'Precision': result['precision'],
                'Recall': result['recall'],
                'F1-Score': result['f1_score'],
                'CV_Mean': result['cv_mean'],
                'CV_Std': result['cv_std']
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        df_comparison.to_csv(f'{OUTPUT_DIR}/model_comparison.csv', index=False)
        
        # Detailed classification reports
        for name, result in self.results.items():
            report = classification_report(
                self.y_test, 
                result['y_pred'],
                target_names=self.label_encoder.classes_,
                output_dict=True
            )
            df_report = pd.DataFrame(report).transpose()
            df_report.to_csv(f'{OUTPUT_DIR}/{name}_classification_report.csv')
        
        print(f"  ✓ Results saved in '{OUTPUT_DIR}/' directory")
    
    def plot_model_comparison(self):
        """Plot comparison of all models"""
        print("Generating model comparison plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        models = [name.replace('_', ' ') for name in self.results.keys()]
        model_keys = list(self.results.keys())
        
        # Accuracy comparison
        accuracies = [self.results[m]['accuracy'] for m in model_keys]
        axes[0, 0].bar(models, accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Model Accuracy Comparison')
        axes[0, 0].set_ylim([0, 1])
        axes[0, 0].tick_params(axis='x', rotation=15)
        for i, v in enumerate(accuracies):
            axes[0, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
        
        # Precision, Recall, F1-Score comparison
        metrics = ['precision', 'recall', 'f1_score']
        x = np.arange(len(models))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            values = [self.results[m][metric] for m in model_keys]
            axes[0, 1].bar(x + i*width, values, width, 
                          label=metric.replace('_', '-').title())
        
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_title('Precision, Recall, F1-Score Comparison')
        axes[0, 1].set_xticks(x + width)
        axes[0, 1].set_xticklabels(models, rotation=15)
        axes[0, 1].legend()
        axes[0, 1].set_ylim([0, 1])
        
        # Cross-validation scores with error bars
        cv_means = [self.results[m]['cv_mean'] for m in model_keys]
        cv_stds = [self.results[m]['cv_std'] for m in model_keys]
        axes[1, 0].bar(models, cv_means, yerr=cv_stds, capsize=5, 
                       color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
        axes[1, 0].set_ylabel('Cross-Validation Score')
        axes[1, 0].set_title('5-Fold Cross-Validation Results')
        axes[1, 0].set_ylim([0, 1])
        axes[1, 0].tick_params(axis='x', rotation=15)
        
        # Best model highlight
        best_model_key = max(self.results.items(), key=lambda x: x[1]['accuracy'])[0]
        best_model_name = best_model_key.replace('_', ' ')
        best_acc = self.results[best_model_key]['accuracy']
        
        axes[1, 1].axis('off')
        axes[1, 1].text(0.5, 0.6, 'Best Model', ha='center', va='center', 
                       fontsize=20, fontweight='bold')
        axes[1, 1].text(0.5, 0.4, best_model_name, ha='center', va='center', 
                       fontsize=24, color='green', fontweight='bold')
        axes[1, 1].text(0.5, 0.2, f'Accuracy: {best_acc:.4f}', ha='center', 
                       va='center', fontsize=16)
        
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/model_comparison.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: model_comparison.png")
        plt.close()
    
    def plot_confusion_matrices(self):
        """Plot confusion matrices for all models"""
        print("Generating confusion matrices...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        axes = axes.ravel()
        
        for idx, (name, result) in enumerate(self.results.items()):
            cm = result['confusion_matrix']
            
            # Plot confusion matrix
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=self.label_encoder.classes_,
                       yticklabels=self.label_encoder.classes_,
                       ax=axes[idx], cbar_kws={'label': 'Count'})
            
            display_name = name.replace('_', ' ')
            axes[idx].set_title(f'{display_name} - Confusion Matrix\nAccuracy: {result["accuracy"]:.4f}')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/confusion_matrices.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: confusion_matrices.png")
        plt.close()
    
    def plot_feature_importance(self):
        """Plot feature importance for Random Forest"""
        print("Generating feature importance plot...")
        
        rf_model = self.results['Random_Forest']['model']
        feature_names = ['Red', 'Green', 'Blue', 'Clear', 'Distance']
        importances = rf_model.feature_importances_
        
        # Sort by importance
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(importances)), importances[indices], 
                color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#96CEB4'])
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices])
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.title('Random Forest - Feature Importance')
        plt.grid(axis='y', alpha=0.3)
        
        # Add values on bars
        for i, v in enumerate(importances[indices]):
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/feature_importance.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: feature_importance.png")
        plt.close()
    
    def plot_learning_curves(self):
        """Plot learning curves for Neural Network"""
        print("Generating learning curves...")
        
        nn_model = self.results['Neural_Network']['model']
        
        if hasattr(nn_model, 'loss_curve_'):
            plt.figure(figsize=(10, 6))
            plt.plot(nn_model.loss_curve_, linewidth=2)
            plt.xlabel('Iterations')
            plt.ylabel('Loss')
            plt.title('Neural Network - Training Loss Curve')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{OUTPUT_DIR}/nn_learning_curve.png', dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved: nn_learning_curve.png")
            plt.close()
    
    def plot_per_class_performance(self):
        """Plot per-class performance for all models"""
        print("Generating per-class performance plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.ravel()
        
        for idx, (name, result) in enumerate(self.results.items()):
            # Get classification report
            report = classification_report(
                self.y_test, 
                result['y_pred'],
                target_names=self.label_encoder.classes_,
                output_dict=True
            )
            
            # Extract per-class metrics
            classes = self.label_encoder.classes_
            precision = [report[c]['precision'] for c in classes]
            recall = [report[c]['recall'] for c in classes]
            f1 = [report[c]['f1-score'] for c in classes]
            
            x = np.arange(len(classes))
            width = 0.25
            
            axes[idx].bar(x - width, precision, width, label='Precision', alpha=0.8)
            axes[idx].bar(x, recall, width, label='Recall', alpha=0.8)
            axes[idx].bar(x + width, f1, width, label='F1-Score', alpha=0.8)
            
            axes[idx].set_xlabel('Classes')
            axes[idx].set_ylabel('Score')
            display_name = name.replace('_', ' ')
            axes[idx].set_title(f'{display_name} - Per-Class Performance')
            axes[idx].set_xticks(x)
            axes[idx].set_xticklabels(classes, rotation=45)
            axes[idx].legend()
            axes[idx].set_ylim([0, 1.1])
            axes[idx].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/per_class_performance.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: per_class_performance.png")
        plt.close()
    
    def generate_summary_report(self):
        """Generate a text summary report"""
        print("\nGenerating summary report...")
        
        with open(f'{OUTPUT_DIR}/summary_report.txt', 'w') as f:
            f.write("="*70 + "\n")
            f.write("COLOR CLASSIFICATION - MODEL PERFORMANCE SUMMARY\n")
            f.write("="*70 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset: {self.data_file}\n")
            f.write(f"Training samples: {len(self.X_train)}\n")
            f.write(f"Testing samples: {len(self.X_test)}\n")
            f.write(f"Classes: {', '.join(self.label_encoder.classes_)}\n")
            f.write("="*70 + "\n\n")
            
            # Model rankings
            f.write("MODEL RANKINGS (by Accuracy):\n")
            f.write("-"*70 + "\n")
            
            ranked = sorted(self.results.items(), 
                          key=lambda x: x[1]['accuracy'], reverse=True)
            
            for rank, (name, result) in enumerate(ranked, 1):
                display_name = name.replace('_', ' ')
                f.write(f"{rank}. {display_name:20s} - Accuracy: {result['accuracy']:.4f}\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("DETAILED METRICS\n")
            f.write("="*70 + "\n\n")
            
            for name, result in self.results.items():
                display_name = name.replace('_', ' ')
                f.write(f"{display_name}:\n")
                f.write(f"  Accuracy:  {result['accuracy']:.4f}\n")
                f.write(f"  Precision: {result['precision']:.4f}\n")
                f.write(f"  Recall:    {result['recall']:.4f}\n")
                f.write(f"  F1-Score:  {result['f1_score']:.4f}\n")
                f.write(f"  CV Score:  {result['cv_mean']:.4f} (+/- {result['cv_std']:.4f})\n")
                f.write("\n")
            
            f.write("="*70 + "\n")
            f.write("SAVED MODELS\n")
            f.write("="*70 + "\n")
            f.write(f"Location: {MODELS_DIR}/\n")
            f.write("Files:\n")
            for name in self.results.keys():
                f.write(f"  - {name}.pkl\n")
            f.write("  - scaler.pkl\n")
            f.write("  - label_encoder.pkl\n")
            f.write("  - model_info.json\n")
            f.write("="*70 + "\n")
        
        print(f"  ✓ Saved: summary_report.txt")
    
    def run_complete_analysis(self):
        """Run complete training and analysis pipeline"""
        
        # Load data
        df = self.load_and_prepare_data()
        
        # Train models
        self.train_models()
        
        # Save models
        self.save_models()
        
        # Save results
        self.save_results_csv()
        
        # Generate plots
        print("\n" + "="*70)
        print("GENERATING VISUALIZATIONS")
        print("="*70)
        self.plot_model_comparison()
        self.plot_confusion_matrices()
        self.plot_feature_importance()
        self.plot_learning_curves()
        self.plot_per_class_performance()
        print("="*70 + "\n")
        
        # Generate summary
        self.generate_summary_report()
        
        # Final summary
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE!")
        print("="*70)
        print(f"Results saved in: '{OUTPUT_DIR}/'")
        print(f"Models saved in: '{MODELS_DIR}/'")
        print("\nGenerated files:")
        print(f"\n{OUTPUT_DIR}/:")
        for file in os.listdir(OUTPUT_DIR):
            print(f"  - {file}")
        print(f"\n{MODELS_DIR}/:")
        for file in os.listdir(MODELS_DIR):
            print(f"  - {file}")
        print("="*70)

def main():
    """Main execution"""
    print("\n" + "="*70)
    print("COLOR CLASSIFICATION - ML TRAINING & ANALYSIS")
    print("="*70 + "\n")
    
    if not os.path.exists(CSV_FILE):
        print(f"Error: Dataset file '{CSV_FILE}' not found!")
        print("\nAvailable CSV files:")
        for file in os.listdir('.'):
            if file.endswith('.csv'):
                print(f"  - {file}")
        return
    
    # Create classifier and run analysis
    classifier = ColorClassifier(CSV_FILE)
    classifier.run_complete_analysis()

if __name__ == "__main__":
    main()