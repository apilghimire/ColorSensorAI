import serial
import pickle
import json
import numpy as np
import os
import time
from datetime import datetime
from collections import Counter

# ======= CONFIGURATION =======
SERIAL_PORT = '/dev/cu.wchusbserial58FA0447301'  # Change to your port
BAUD_RATE = 115200
MODELS_DIR = 'models'


# Choose which model to use for prediction
# Options: 'SVM', 'Random_Forest', 'KNN', 'Neural_Network', 'ALL' (ensemble voting)
PREDICTION_MODEL = 'KNN'

# Confidence threshold (predictions below this are marked as "Unspecified")
CONFIDENCE_THRESHOLD = 0.80  # 80%

class RealtimeColorPredictor:
    """Real-time color prediction from ESP32-S3 sensor data"""
    
    def __init__(self, models_dir):
        self.models_dir = models_dir
        self.models = {}
        self.scaler = None
        self.label_encoder = None
        self.model_info = None
        self.load_models()
    
    def load_models(self):
        """Load all saved models and preprocessing objects"""
        print("="*70)
        print("LOADING MODELS")
        print("="*70)
        
        # Check if models directory exists
        if not os.path.exists(self.models_dir):
            raise FileNotFoundError(f"Models directory '{self.models_dir}' not found!")
        
        # Load model info
        info_path = os.path.join(self.models_dir, 'model_info.json')
        if not os.path.exists(info_path):
            raise FileNotFoundError("model_info.json not found! Please train models first.")
        
        with open(info_path, 'r') as f:
            self.model_info = json.load(f)
        
        print(f"Model trained on: {self.model_info['training_date']}")
        print(f"Classes: {', '.join(self.model_info['classes'])}")
        print(f"Total training samples: {self.model_info['total_samples']}\n")
        
        # Load scaler
        scaler_path = os.path.join(self.models_dir, 'scaler.pkl')
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        print("✓ Loaded: scaler.pkl")
        
        # Load label encoder
        encoder_path = os.path.join(self.models_dir, 'label_encoder.pkl')
        with open(encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        print("✓ Loaded: label_encoder.pkl")
        
        # Load all models
        model_files = ['SVM.pkl', 'Random_Forest.pkl', 'KNN.pkl', 'Neural_Network.pkl']
        
        for model_file in model_files:
            model_path = os.path.join(self.models_dir, model_file)
            if os.path.exists(model_path):
                model_name = model_file.replace('.pkl', '')
                with open(model_path, 'rb') as f:
                    self.models[model_name] = pickle.load(f)
                
                # Get accuracy from metadata
                accuracy = self.model_info['model_performance'][model_name]['accuracy']
                print(f"✓ Loaded: {model_file} (Accuracy: {accuracy:.4f})")
        
        print("\n" + "="*70 + "\n")
        
        if not self.models:
            raise ValueError("No models loaded! Please train models first.")
    
    def preprocess_data(self, raw_data):
        """Preprocess raw sensor data"""
        # raw_data is [R, G, B, C, Distance]
        data_array = np.array(raw_data).reshape(1, -1)
        scaled_data = self.scaler.transform(data_array)
        return scaled_data
    
    def predict_single_model(self, model_name, data):
        """Predict using a single model"""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not loaded!")
        
        model = self.models[model_name]
        prediction = model.predict(data)[0]
        label = self.label_encoder.inverse_transform([prediction])[0]
        
        # Get probability if available
        probability = None
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(data)[0]
            probability = probs[prediction]
            
            # Check confidence threshold
            if probability < 0.80:
                label = "Unspecified"
        
        return label, probability
    
    def predict_ensemble(self, data):
        """Predict using all models (voting ensemble)"""
        predictions = []
        probabilities = []
        
        for model_name, model in self.models.items():
            pred = model.predict(data)[0]
            predictions.append(pred)
            
            # Get probability if available
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(data)[0]
                probabilities.append(probs[pred])
        
        # Majority voting
        vote_counts = Counter(predictions)
        most_common = vote_counts.most_common(1)[0]
        final_prediction = most_common[0]
        votes = most_common[1]
        
        label = self.label_encoder.inverse_transform([final_prediction])[0]
        avg_confidence = np.mean(probabilities) if probabilities else None
        
        # Check confidence threshold
        if avg_confidence is not None and avg_confidence < 0.80:
            label = "Unspecified"
        
        return label, avg_confidence, votes, len(predictions)
    
    def predict(self, raw_data):
        """Main prediction function"""
        # Preprocess data
        processed_data = self.preprocess_data(raw_data)
        
        if PREDICTION_MODEL == 'ALL':
            # Ensemble prediction
            label, confidence, votes, total = self.predict_ensemble(processed_data)
            return {
                'label': label,
                'confidence': confidence,
                'votes': f"{votes}/{total}",
                'method': 'Ensemble'
            }
        else:
            # Single model prediction
            label, confidence = self.predict_single_model(PREDICTION_MODEL, processed_data)
            return {
                'label': label,
                'confidence': confidence,
                'method': PREDICTION_MODEL
            }
    
    def run_realtime_prediction(self, serial_port, baud_rate):
        """Run real-time prediction from serial data"""
        print("="*70)
        print("REAL-TIME COLOR PREDICTION")
        print("="*70)
        print(f"Prediction mode: {PREDICTION_MODEL}")
        print(f"Confidence threshold: {CONFIDENCE_THRESHOLD*100:.0f}%")
        print(f"Serial port: {serial_port}")
        print(f"Baud rate: {baud_rate}")
        print("\nPress Ctrl+C to stop\n")
        print("="*70)
        
        try:
            # Open serial connection
            print(f"\nConnecting to {serial_port}...")
            ser = serial.Serial(serial_port, baud_rate, timeout=1)
            time.sleep(2)  # Wait for connection
            
            # Clear buffer
            ser.reset_input_buffer()
            print("✓ Connected!\n")
            
            print("-"*70)
            print(f"{'Time':<12} {'R':>5} {'G':>5} {'B':>5} {'C':>6} {'Dist':>5} | {'Prediction':<15} {'Confidence':<12}")
            print("-"*70)
            
            prediction_count = 0
            
            while True:
                try:
                    if ser.in_waiting > 0:
                        line = ser.readline().decode('utf-8').strip()
                        
                        # Skip empty lines or error messages
                        if not line or 'NOT FOUND' in line:
                            continue
                        
                        # Parse CSV data: R,G,B,C,Distance
                        data = line.split(',')
                        
                        if len(data) == 5:
                            try:
                                # Convert to integers
                                sensor_data = [int(x) for x in data]
                                r, g, b, c, dist = sensor_data
                                
                                # Make prediction
                                result = self.predict(sensor_data)
                                
                                # Format output
                                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                                
                                confidence_str = ""
                                if result['confidence'] is not None:
                                    confidence_str = f"{result['confidence']:.2%}"
                                else:
                                    confidence_str = "N/A"
                                
                                if 'votes' in result:
                                    confidence_str += f" ({result['votes']})"
                                
                                # Color the prediction based on confidence
                                prediction_count += 1
                                
                                print(f"{timestamp} {r:5d} {g:5d} {b:5d} {c:6d} {dist:5d} | "
                                      f"{result['label']:<15} {confidence_str:<12}")
                                
                            except ValueError:
                                # Invalid data format
                                continue
                
                except UnicodeDecodeError:
                    # Skip lines with encoding errors
                    continue
        
        except serial.SerialException as e:
            print(f"\n✗ Serial Error: {e}")
            print("Please check:")
            print("1. Correct port name")
            print("2. ESP32 is connected")
            print("3. No other program is using the port")
        
        except KeyboardInterrupt:
            print(f"\n\n{'='*70}")
            print(f"Stopped after {prediction_count} predictions")
            print(f"{'='*70}")
        
        finally:
            if 'ser' in locals() and ser.is_open:
                ser.close()
                print("Serial port closed")

def show_model_performance():
    """Display performance of all available models"""
    info_path = os.path.join(MODELS_DIR, 'model_info.json')
    
    if not os.path.exists(info_path):
        print("No model info found. Please train models first.")
        return
    
    with open(info_path, 'r') as f:
        info = json.load(f)
    
    print("\n" + "="*70)
    print("AVAILABLE MODELS PERFORMANCE")
    print("="*70)
    print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-"*70)
    
    for model_name, metrics in info['model_performance'].items():
        display_name = model_name.replace('_', ' ')
        print(f"{display_name:<20} {metrics['accuracy']:<12.4f} "
              f"{metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
              f"{metrics['f1_score']:<12.4f}")
    
    print("="*70 + "\n")

def main():
    """Main execution"""
    print("\n" + "="*70)
    print("REAL-TIME COLOR CLASSIFICATION")
    print("="*70 + "\n")
    
    # Show model performance
    show_model_performance()
    
    # Create predictor
    try:
        predictor = RealtimeColorPredictor(MODELS_DIR)
    except Exception as e:
        print(f"Error loading models: {e}")
        print("\nPlease run the training script first to generate models!")
        return
    
    # Ask for prediction mode if not set
    global PREDICTION_MODEL
    if PREDICTION_MODEL == 'ALL':
        print("Prediction mode: ENSEMBLE (All models voting)")
    else:
        print(f"Prediction mode: {PREDICTION_MODEL}")
    
    print(f"\nTo change prediction mode, edit PREDICTION_MODEL in the script")
    print("Options: 'SVM', 'Random_Forest', 'KNN', 'Neural_Network', 'ALL'\n")
    
    response = input("Start real-time prediction? (y/n): ").strip().lower()
    
    if response == 'y':
        predictor.run_realtime_prediction(SERIAL_PORT, BAUD_RATE)
    else:
        print("Exiting...")

if __name__ == "__main__":
    main()