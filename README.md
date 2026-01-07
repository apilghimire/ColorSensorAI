# ColorSensorAI

A machine learning project for real-time color classification using RGB sensor data from Arduino hardware. This project compares three different machine learning algorithms to classify colors based on sensor readings.

## Project Overview

This project uses RGB color sensor data collected from an Arduino device to train and compare multiple machine learning models for accurate color classification. The system converts RGB values to the CIELAB color space for improved color discrimination and trains three different classifiers to identify colors in real-time.

## Features

- **Real-time Color Detection**: Live classification of colors from Arduino RGB sensor
- **Multiple ML Algorithms**: Comparison of K-Nearest Neighbors (KNN), Support Vector Machine (SVM), and Random Forest classifiers
- **CIELAB Color Space Conversion**: RGB values are converted to CIELAB for perceptually uniform color representation
- **Data Collection**: Arduino-based data retrieval system for building custom color datasets
- **Visualization**: Confusion matrices and feature importance plots for model evaluation

## Dataset

The dataset consists of RGB sensor readings with distance measurements for the following colors:
- Red (~531 samples)
- Green (~440 samples)
- Blue (~544 samples)
- Yellow (~518 samples)
- White (~506 samples)
- TooFar (~543 samples - object out of range)

**Total Dataset**: ~3,077 samples

Each sample contains:
- Red intensity value
- Green intensity value
- Blue intensity value
- Distance measurement (mm)
- Color label

## Models and Results

### 1. K-Nearest Neighbors (KNN)
- **Algorithm**: K-Nearest Neighbors with distance weighting
- **Parameters**: k=5, metric='euclidean', weights='distance'
- **Features**: RGB → CIELAB conversion with standard scaling
- **Output**: 
  - Trained model: `knn_model.pkl`
  - Confusion matrix: `knn_confusion_matrix.png`

### 2. Random Forest (RF)
- **Algorithm**: Random Forest Classifier with GridSearchCV optimization
- **Features**: RGB → CIELAB conversion with standard scaling
- **Feature Importance**: Analyzes which color channels contribute most to classification
- **Output**: 
  - Trained model: `rf_model.pkl`
  - Confusion matrix: `rf_confusion_matrix.png`
  - Feature importance plot: `rf_feature_importance.png`

### 3. Support Vector Machine (SVM)
- **Algorithm**: SVM with RBF kernel and GridSearchCV optimization
- **Features**: RGB → CIELAB conversion with standard scaling
- **Output**: 
  - Trained model: `svm_model.pkl`
  - Confusion matrix: `svm_confusion_matrix.png`

## Project Structure

```
ColorSensorAI/
├── KNN.py                    # K-Nearest Neighbors classifier
├── RF.py                     # Random Forest classifier
├── SVM.py                    # Support Vector Machine classifier
├── RealTimeModelTester.py    # Real-time testing with all models
├── RetreveData.py            # Arduino data collection script
├── Merge Labels.py           # Dataset preparation utility
├── modelTaster.py            # Model comparison utility
├── plot.py                   # Data visualization
├── data.csv                  # Combined dataset
├── Red.csv                   # Red color samples
├── Green.csv                 # Green color samples
├── Blue.csv                  # Blue color samples
├── Yellow.csv                # Yellow color samples
├── White.csv                 # White color samples
├── tooFar.csv               # Out-of-range samples
└── LICENSE                   # Project license
```

## Technologies Used

- **Python 3.x**
- **scikit-learn**: Machine learning algorithms and preprocessing
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **matplotlib**: Data visualization
- **seaborn**: Statistical visualizations
- **joblib**: Model serialization
- **pyserial**: Arduino serial communication
- **colorama**: Colored terminal output

## How It Works

1. **Data Collection**: 
   - RGB sensor data is collected from Arduino via serial communication
   - Each color is sampled multiple times at various distances
   - Data is stored in separate CSV files by color

2. **Preprocessing**:
   - RGB values are normalized
   - Converted to CIELAB color space for perceptual uniformity
   - Features are standardized using StandardScaler

3. **Model Training**:
   - Data is split into training and testing sets
   - Three different models are trained and optimized
   - Models are evaluated using accuracy metrics and confusion matrices

4. **Real-time Classification**:
   - Live RGB data is received from Arduino sensor
   - All three models make predictions simultaneously
   - Results are displayed with color-coded output

## Setup and Usage

### Requirements
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib pyserial colorama
```

### Training Models
```bash
# Train K-Nearest Neighbors
python KNN.py

# Train Random Forest
python RF.py

# Train Support Vector Machine
python SVM.py
```

### Real-time Testing
```bash
# Run real-time classification with all models
python RealTimeModelTester.py
```

### Data Collection
```bash
# Collect new color samples
python RetreveData.py
```

**Note**: Update the `SERIAL_PORT` variable in `RetreveData.py` and `RealTimeModelTester.py` to match your Arduino's port.

## Hardware Requirements

- Arduino board (e.g., Arduino Uno, Nano)
- RGB color sensor (e.g., TCS3200, TCS34725)
- USB cable for Arduino connection
- Colored objects for testing

## Model Performance

All three models achieve high accuracy in color classification thanks to:
- CIELAB color space conversion
- Proper feature scaling
- Distance-based measurements for improved discrimination
- Hyperparameter tuning via GridSearchCV (RF and SVM)

For detailed classification reports and confusion matrices, run the individual model training scripts.

## Future Improvements

- Add more color categories
- Implement deep learning models (CNN)
- Create a web interface for visualization
- Add calibration routine for different lighting conditions
- Expand dataset with more samples per color
- Implement ensemble methods combining all three models

## License

This project is licensed under the terms specified in the LICENSE file.

## Author

Apil Ghimire

---

*This project demonstrates practical application of machine learning for embedded systems and IoT applications.*
