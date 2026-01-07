import serial
import numpy as np
import joblib
import time
from datetime import datetime
from colorama import init, Fore, Back, Style

# Initialize colorama for colored terminal output
init(autoreset=True)

# Configuration
SERIAL_PORT = '/dev/cu.wchusbserial14140'  # Change this to your Arduino port
BAUD_RATE = 9600

# Model files
MODELS = {
    'KNN': {
        'model': 'knn_model.pkl',
        'scaler': 'scaler.pkl',
        'norm_params': 'norm_params.pkl',
        'color': Fore.CYAN
    },
    'SVM': {
        'model': 'svm_model.pkl',
        'scaler': 'svm_scaler.pkl',
        'norm_params': 'svm_norm_params.pkl',
        'color': Fore.GREEN
    },
    'Random Forest': {
        'model': 'rf_model.pkl',
        'scaler': 'rf_scaler.pkl',
        'norm_params': 'rf_norm_params.pkl',
        'color': Fore.YELLOW
    }
}


def rgb_to_cielab(r, g, b):
    """Convert RGB to CIELAB color space"""
    r, g, b = r / 255.0, g / 255.0, b / 255.0

    def gamma_correction(channel):
        if channel > 0.04045:
            return ((channel + 0.055) / 1.055) ** 2.4
        else:
            return channel / 12.92

    r, g, b = gamma_correction(r), gamma_correction(g), gamma_correction(b)

    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

    x, y, z = x / 0.95047, y / 1.00000, z / 1.08883

    def f(t):
        if t > 0.008856:
            return t ** (1 / 3)
        else:
            return (7.787 * t) + (16 / 116)

    l = 116 * f(y) - 16
    a = 500 * (f(x) - f(y))
    b_lab = 200 * (f(y) - f(z))

    return l, a, b_lab


def load_all_models():
    """Load all trained models and preprocessing parameters"""
    loaded_models = {}

    print("Loading models...")
    for model_name, files in MODELS.items():
        try:
            model = joblib.load(files['model'])
            scaler = joblib.load(files['scaler'])
            norm_params = joblib.load(files['norm_params'])

            loaded_models[model_name] = {
                'model': model,
                'scaler': scaler,
                'norm_params': norm_params,
                'color': files['color'],
                'correct': 0,
                'total': 0
            }
            print(f"  ✓ {model_name} loaded successfully")

        except FileNotFoundError as e:
            print(f"  ✗ {model_name} - Missing file: {e.filename}")
            print(f"    Please train {model_name} first")

    if not loaded_models:
        print("\nError: No models loaded!")
        print("Please train at least one model first.")
        exit(1)

    return loaded_models


def preprocess_sensor_data(red, green, blue, distance, norm_params):
    """Preprocess sensor data to match training format"""
    # Normalize RGB values
    r_norm = (red / norm_params['max_red']) * 255 if red > 0 else 0
    g_norm = (green / norm_params['max_green']) * 255 if green > 0 else 0
    b_norm = (blue / norm_params['max_blue']) * 255 if blue > 0 else 0

    # Convert to CIELAB
    l, a, b = rgb_to_cielab(r_norm, g_norm, b_norm)

    # Compute intensity
    intensity = (red + green + blue) / 3

    # Create feature vector [L, a, b, intensity, distance]
    feature_vector = np.array([[l, a, b, intensity, distance]])

    return feature_vector


def predict_with_all_models(loaded_models, red, green, blue, distance):
    """Make predictions with all models"""
    predictions = {}

    for model_name, model_data in loaded_models.items():
        # Preprocess data
        feature_vector = preprocess_sensor_data(
            red, green, blue, distance, model_data['norm_params']
        )

        # Scale features
        feature_scaled = model_data['scaler'].transform(feature_vector)

        # Predict
        prediction = model_data['model'].predict(feature_scaled)[0]

        # Get confidence if available
        if hasattr(model_data['model'], 'predict_proba'):
            probabilities = model_data['model'].predict_proba(feature_scaled)[0]
            confidence = max(probabilities) * 100
        else:
            confidence = None

        predictions[model_name] = {
            'prediction': prediction,
            'confidence': confidence,
            'color': model_data['color']
        }

    return predictions


def display_predictions(timestamp, pred_num, red, green, blue, distance, predictions, actual_label=None):
    """Display predictions from all models in a formatted way"""
    print(f"\n{Style.BRIGHT}{'=' * 80}")
    print(f"[{timestamp}] Prediction #{pred_num:04d}")
    print(f"Sensor Data: R:{red:4d} G:{green:4d} B:{blue:4d} Distance:{distance:4d}mm")
    print(f"{'=' * 80}{Style.RESET_ALL}")

    # Check if all models agree
    all_predictions = [p['prediction'] for p in predictions.values()]
    all_agree = len(set(all_predictions)) == 1

    if all_agree:
        print(f"{Back.GREEN}{Fore.BLACK} ✓ ALL MODELS AGREE {Style.RESET_ALL}")
    else:
        print(f"{Back.RED}{Fore.WHITE} ⚠ MODELS DISAGREE {Style.RESET_ALL}")

    print()

    # Display each model's prediction
    for model_name, pred_data in predictions.items():
        color = pred_data['color']
        prediction = pred_data['prediction']
        confidence = pred_data['confidence']

        # Highlight if actual label is provided
        match_indicator = ""
        if actual_label:
            if prediction == actual_label:
                match_indicator = f"{Back.GREEN}{Fore.BLACK} ✓ CORRECT {Style.RESET_ALL}"
            else:
                match_indicator = f"{Back.RED}{Fore.WHITE} ✗ WRONG {Style.RESET_ALL}"

        if confidence:
            print(f"{color}{Style.BRIGHT}{model_name:15s}{Style.RESET_ALL} → "
                  f"{Style.BRIGHT}{prediction:10s}{Style.RESET_ALL} "
                  f"(Confidence: {confidence:5.1f}%) {match_indicator}")
        else:
            print(f"{color}{Style.BRIGHT}{model_name:15s}{Style.RESET_ALL} → "
                  f"{Style.BRIGHT}{prediction:10s}{Style.RESET_ALL} {match_indicator}")

    if actual_label:
        print(f"\n{Style.BRIGHT}Actual Label: {actual_label}{Style.RESET_ALL}")


def update_accuracy(loaded_models, predictions, actual_label):
    """Update accuracy statistics for each model"""
    for model_name, pred_data in predictions.items():
        loaded_models[model_name]['total'] += 1
        if pred_data['prediction'] == actual_label:
            loaded_models[model_name]['correct'] += 1


def display_accuracy_stats(loaded_models):
    """Display current accuracy statistics for all models"""
    print(f"\n{Style.BRIGHT}{'=' * 80}")
    print("REAL-TIME ACCURACY STATISTICS")
    print(f"{'=' * 80}{Style.RESET_ALL}")

    for model_name, model_data in loaded_models.items():
        total = model_data['total']
        correct = model_data['correct']
        accuracy = (correct / total * 100) if total > 0 else 0

        color = model_data['color']
        print(f"{color}{Style.BRIGHT}{model_name:15s}{Style.RESET_ALL}: "
              f"{correct}/{total} correct = {accuracy:5.1f}% accuracy")


def main():
    print(f"{Style.BRIGHT}{'=' * 80}")
    print("REAL-TIME COLOR CLASSIFICATION - ALL MODELS COMPARISON")
    print(f"{'=' * 80}{Style.RESET_ALL}\n")

    # Load all models
    loaded_models = load_all_models()

    print(f"\n{Style.BRIGHT}Loaded {len(loaded_models)} model(s){Style.RESET_ALL}\n")

    # Connect to Arduino
    print(f"Connecting to Arduino on {SERIAL_PORT}...")

    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)  # Wait for Arduino to reset

        print(f"{Fore.GREEN}✓ Connected to Arduino!{Style.RESET_ALL}\n")
        print(f"{Style.BRIGHT}Starting real-time predictions...{Style.RESET_ALL}")
        print("Press Ctrl+C to stop\n")
        print("Options:")
        print("  - Just watch predictions update automatically")
        print("  - Or type the actual color after each prediction to track accuracy")
        print(f"{'=' * 80}\n")

        # Skip initialization messages
        while True:
            line = ser.readline().decode('utf-8').strip()
            if line and not line.startswith('Initializing') and not line.startswith('Sensors'):
                break

        # Main prediction loop
        prediction_count = 0

        while True:
            try:
                # Read line from serial
                line = ser.readline().decode('utf-8').strip()

                if line:
                    # Parse comma-separated values
                    values = line.split(',')

                    if len(values) == 4:
                        try:
                            red = int(values[0])
                            green = int(values[1])
                            blue = int(values[2])
                            distance = int(values[3])

                            prediction_count += 1
                            timestamp = datetime.now().strftime('%H:%M:%S')

                            # Make predictions with all models
                            predictions = predict_with_all_models(
                                loaded_models, red, green, blue, distance
                            )

                            # Display predictions
                            display_predictions(
                                timestamp, prediction_count, red, green, blue,
                                distance, predictions
                            )

                            # Optional: Ask for actual label to track accuracy
                            # Uncomment the lines below if you want to manually verify
                            # print("\nEnter actual color (or press Enter to skip): ", end='')
                            # actual_label = input().strip()
                            # if actual_label:
                            #     update_accuracy(loaded_models, predictions, actual_label)
                            #     display_accuracy_stats(loaded_models)

                        except ValueError:
                            print(f"Warning: Invalid data format: {line}")
                            continue

            except KeyboardInterrupt:
                print(f"\n\n{Style.BRIGHT}{'=' * 80}")
                print(f"Stopped! Total predictions made: {prediction_count}")
                display_accuracy_stats(loaded_models)
                print(f"{'=' * 80}{Style.RESET_ALL}")
                break
            except Exception as e:
                print(f"Error: {e}")
                continue

        ser.close()
        print("\nSerial connection closed.")

    except serial.SerialException as e:
        print(f"{Fore.RED}Error: Could not open serial port {SERIAL_PORT}{Style.RESET_ALL}")
        print(f"Details: {e}")
        print("\nTips:")
        print("- Check if the Arduino is connected")
        print("- Verify the correct COM port")
        print("- Close Arduino IDE Serial Monitor if open")
        print("- On Linux, you may need: sudo chmod 666 /dev/ttyUSB0")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()