# ColorSensorAI

Color classification system using ESP32-S3 with TCS34725 color sensor and VL53L0X distance sensor.

## Virtual Environment Setup

### Activating the Virtual Environment

1. On macOS/Linux:
   ```bash
   source .venv/bin/activate
   ```

2. On Windows:
   ```bash
   .venv\Scripts\activate
   ```

3. To deactivate when done:
   ```bash
   deactivate
   ```

## Required Dependencies

Install required packages after activating the virtual environment:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn pyserial
```

## Available Scripts

### 1. retreveData.py

**Purpose:** Collects raw sensor data from ESP32-S3 via serial connection and saves it to CSV files.

**How it works:**
1. Establishes serial connection with ESP32-S3 at specified port and baud rate
2. Reads sensor data (Red, Green, Blue, Clear, Distance, Timestamp) from the microcontroller
3. Saves data in real-time to a CSV file with specified filename
4. Continues collecting until MAX_SAMPLES is reached or manually stopped

**Configuration:**
- SERIAL_PORT: Set to your ESP32-S3 serial port (e.g., '/dev/cu.wchusbserial58FA0447301')
- BAUD_RATE: Communication speed (default: 115200)
- CSV_FILENAME: Output file name (e.g., 'Red.csv', 'Blue.csv')
- MAX_SAMPLES: Number of samples to collect (default: 600)

**Usage:**
```bash
python retreveData.py
```

**Output:** CSV file with sensor readings for a single color/object.

---

### 2. plot.py

**Purpose:** Visualizes sensor data from CSV files to analyze patterns and distributions.

**How it works:**
1. Loads sensor data from specified CSV file
2. Calculates statistical metrics (mean, median, std, min, max) for each sensor
3. Generates multiple visualization plots:
   - Time series plots showing sensor values over time
   - Distribution histograms for each sensor
   - Correlation heatmap between sensors
   - Box plots showing data spread and outliers
4. Saves plots as PNG files

**Configuration:**
- CSV_FILE: Input CSV file to analyze (e.g., 'Red.csv', 'Blue.csv')

**Usage:**
```bash
python plot.py
```

**Output:** 
- Statistical summary printed to console
- PNG files with analysis plots (e.g., 'Red_analysis.png', 'Red_correlation.png')

---

### 3. mergeLabels.py

**Purpose:** Combines multiple labeled CSV files into a single dataset for machine learning.

**How it works:**
1. Reads multiple CSV files (one per color/class)
2. Adds a 'Label' column to each dataset identifying the color
3. Merges all datasets into a single DataFrame
4. Removes any duplicate or invalid entries
5. Saves the combined dataset with timestamp

**Configuration:**
- CSV_FILES: Dictionary mapping filenames to their labels
  ```python
  CSV_FILES = {
      'Red.csv': 'Red',
      'Blue.csv': 'Blue',
      'Green.csv': 'Green'
  }
  ```
- OUTPUT_FILE: Name for merged dataset (default: 'Data.csv')

**Usage:**
```bash
python mergeLabels.py
```

**Output:** Single CSV file containing all labeled data ready for model training.

---

### 4. model.py

**Purpose:** Trains machine learning models for color classification and evaluates their performance.

**How it works:**
1. Loads merged dataset from Data.csv
2. Preprocesses data (feature scaling, label encoding)
3. Splits data into training (80%) and testing (20%) sets
4. Trains four different ML models:
   - Support Vector Machine (SVM)
   - Random Forest Classifier
   - K-Nearest Neighbors (KNN)
   - Neural Network (MLP)
5. Evaluates each model using multiple metrics:
   - Accuracy, Precision, Recall, F1-Score
   - Cross-validation scores
   - Confusion matrices
6. Generates comprehensive visualizations:
   - Model comparison charts
   - Confusion matrices for each model
   - Feature importance analysis
   - Learning curves
   - Per-class performance metrics
7. Saves trained models, scalers, and encoders to 'models/' directory
8. Saves all results and reports to 'Results/' directory

**Configuration:**
- CSV_FILE: Input dataset file (default: 'Data.csv')
- TEST_SIZE: Percentage of data for testing (default: 0.2)
- RANDOM_STATE: Seed for reproducibility (default: 42)
- OUTPUT_DIR: Directory for results (default: 'Results')
- MODELS_DIR: Directory for saved models (default: 'models')

**Usage:**
```bash
python model.py
```

**Output:**
- Trained models saved in models/ directory:
  - Individual model files (.pkl)
  - scaler.pkl (StandardScaler)
  - label_encoder.pkl (LabelEncoder)
  - model_info.json (metadata)
- Results in Results/ directory:
  - CSV files with metrics and classification reports
  - PNG files with visualizations
  - summary_report.txt with complete analysis

---

### 5. realTimeTest.py

**Purpose:** Uses trained models to perform real-time color prediction from live sensor data.

**How it works:**
1. Loads trained models from 'models/' directory
2. Establishes serial connection with ESP32-S3
3. Reads sensor data in real-time
4. Preprocesses incoming data (same scaling as training)
5. Makes predictions using selected model or ensemble voting
6. Displays predictions with confidence scores
7. Optionally saves prediction history to CSV
8. Provides performance statistics (processing time, prediction counts)

**Prediction Modes:**
- Single model: Uses one specific model (SVM, Random_Forest, KNN, or Neural_Network)
- Ensemble (ALL): Combines predictions from all models using majority voting

**Configuration:**
- SERIAL_PORT: ESP32-S3 serial port
- BAUD_RATE: Communication speed (default: 115200)
- MODELS_DIR: Directory containing trained models (default: 'models')
- PREDICTION_MODEL: Model to use ('SVM', 'Random_Forest', 'KNN', 'Neural_Network', or 'ALL')
- CONFIDENCE_THRESHOLD: Minimum confidence for valid prediction (default: 0.80)

**Usage:**
```bash
python realTimeTest.py
```

**Output:**
- Real-time predictions displayed in console with:
  - Predicted color
  - Confidence percentage
  - Sensor values
  - Processing time
- Optional CSV file with prediction history
- Summary statistics on exit

---

## Typical Workflow

1. Collect data for each color/object:
   ```bash
   # Modify CSV_FILENAME in retreveData.py for each color
   python retreveData.py  # Collect Red.csv
   python retreveData.py  # Collect Blue.csv
   python retreveData.py  # Collect Green.csv
   ```

2. Visualize and verify data quality:
   ```bash
   # Modify CSV_FILE in plot.py for each dataset
   python plot.py
   ```

3. Merge all datasets:
   ```bash
   # Update CSV_FILES dictionary in mergeLabels.py
   python mergeLabels.py
   ```

4. Train models:
   ```bash
   python model.py
   ```

5. Test real-time predictions:
   ```bash
   # Update PREDICTION_MODEL in realTimeTest.py if needed
   python realTimeTest.py
   ```

## Project Structure

```
ColorSensorAI/
├── .venv/                  # Virtual environment
├── models/                 # Trained ML models
├── Results/                # Training results and visualizations
├── Red.csv                 # Red color training data
├── Blue.csv                # Blue color training data
├── Green.csv               # Green color training data
├── Data.csv                # Merged training dataset
├── retreveData.py          # Data collection script
├── plot.py                 # Data visualization script
├── mergeLabels.py          # Dataset merging script
├── model.py                # Model training script
├── realTimeTest.py         # Real-time prediction script
└── README.md               # This file
```

## Hardware Requirements

1. ESP32-S3 microcontroller
2. TCS34725 color sensor
3. VL53L0X distance sensor
4. USB cable for serial connection

## Notes

1. Ensure the ESP32-S3 is properly connected before running data collection or real-time testing scripts
2. The serial port may change when reconnecting the device - update SERIAL_PORT accordingly
3. Model training requires sufficient data samples (recommended: 500+ samples per class)
4. Higher confidence thresholds in real-time testing reduce false positives but may reject more predictions
5. Ensemble mode (ALL) generally provides more robust predictions but slower processing
