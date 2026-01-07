import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time


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


def prepare_data():
    """Load and preprocess data"""
    df = pd.read_csv('data.csv')

    # Get max values for normalization
    max_red = df['Red'].max()
    max_green = df['Green'].max()
    max_blue = df['Blue'].max()

    features = []
    labels = []

    for idx, row in df.iterrows():
        r_norm = (row['Red'] / max_red) * 255 if row['Red'] > 0 else 0
        g_norm = (row['Green'] / max_green) * 255 if row['Green'] > 0 else 0
        b_norm = (row['Blue'] / max_blue) * 255 if row['Blue'] > 0 else 0

        l, a, b = rgb_to_cielab(r_norm, g_norm, b_norm)
        intensity = (row['Red'] + row['Green'] + row['Blue']) / 3

        feature_vector = [l, a, b, intensity, row['Distance_mm']]
        features.append(feature_vector)
        labels.append(row['Label'])

    return np.array(features), np.array(labels)


def compare_classifiers():
    print("=" * 70)
    print("COMPARING ALL THREE CLASSIFIERS")
    print("=" * 70)

    # Load and prepare data
    X, y = prepare_data()
    print(f"\nDataset: {len(X)} samples, {len(np.unique(y))} classes")
    print(f"Classes: {np.unique(y)}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize models
    models = {
        'KNN': KNeighborsClassifier(n_neighbors=5, weights='distance'),
        'SVM': SVC(kernel='rbf', C=10, gamma='scale'),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }

    results = {}

    print("\n" + "=" * 70)
    print("TRAINING AND EVALUATING MODELS")
    print("=" * 70)

    for name, model in models.items():
        print(f"\n{'=' * 70}")
        print(f"Training {name}...")
        print(f"{'=' * 70}")

        # Train
        start_time = time.time()
        model.fit(X_train_scaled, y_train)
        train_time = time.time() - start_time

        # Test
        start_time = time.time()
        y_pred = model.predict(X_test_scaled)
        test_time = time.time() - start_time

        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)

        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()

        results[name] = {
            'accuracy': accuracy,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'train_time': train_time,
            'test_time': test_time
        }

        print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
        print(f"Cross-Validation: {cv_mean * 100:.2f}% (+/- {cv_std * 100:.2f}%)")
        print(f"Training Time: {train_time:.4f} seconds")
        print(f"Testing Time: {test_time:.4f} seconds")

        print(f"\nClassification Report for {name}:")
        print(classification_report(y_test, y_pred))

    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)

    comparison_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Test Accuracy (%)': [r['accuracy'] * 100 for r in results.values()],
        'CV Accuracy (%)': [r['cv_mean'] * 100 for r in results.values()],
        'CV Std (%)': [r['cv_std'] * 100 for r in results.values()],
        'Train Time (s)': [r['train_time'] for r in results.values()],
        'Test Time (s)': [r['test_time'] for r in results.values()]
    })

    print("\n", comparison_df.to_string(index=False))

    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\n{'=' * 70}")
    print(f"BEST MODEL: {best_model[0]} with {best_model[1]['accuracy'] * 100:.2f}% accuracy")
    print(f"{'=' * 70}")

    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Accuracy comparison
    model_names = list(results.keys())
    accuracies = [results[m]['accuracy'] * 100 for m in model_names]
    cv_accuracies = [results[m]['cv_mean'] * 100 for m in model_names]

    x = np.arange(len(model_names))
    width = 0.35

    axes[0].bar(x - width / 2, accuracies, width, label='Test Accuracy', color='steelblue')
    axes[0].bar(x + width / 2, cv_accuracies, width, label='CV Accuracy', color='coral')
    axes[0].set_xlabel('Model')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Model Accuracy Comparison')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(model_names)
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)

    # Training time comparison
    train_times = [results[m]['train_time'] for m in model_names]
    axes[1].bar(model_names, train_times, color='green', alpha=0.7)
    axes[1].set_xlabel('Model')
    axes[1].set_ylabel('Time (seconds)')
    axes[1].set_title('Training Time Comparison')
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300)
    print("\nComparison plot saved as 'model_comparison.png'")

    return results


if __name__ == "__main__":
    compare_classifiers()