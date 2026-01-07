import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Configuration
CSV_FILENAME = 'Green.csv'


def plot_sensor_data():
    try:
        # Read CSV file
        df = pd.read_csv(CSV_FILENAME)

        print(f"Loaded {len(df)} data points from {CSV_FILENAME}")

        # Convert timestamp to datetime for better x-axis
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])

        # Create figure with subplots
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle('Sensor Data Visualization', fontsize=16, fontweight='bold')

        # Plot 1: Color Sensor Data (RGB)
        axes[0].plot(df['Timestamp'], df['Red'], 'r-', label='Red', linewidth=2)
        axes[0].plot(df['Timestamp'], df['Green'], 'g-', label='Green', linewidth=2)
        axes[0].plot(df['Timestamp'], df['Blue'], 'b-', label='Blue', linewidth=2)
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('TCS3200 Color Sensor - RGB Values')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Distance Sensor Data
        axes[1].plot(df['Timestamp'], df['Distance_mm'], 'purple', linewidth=2)
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Distance (mm)')
        axes[1].set_title('VL53L0X Distance Sensor')
        axes[1].grid(True, alpha=0.3)
        axes[1].fill_between(df['Timestamp'], df['Distance_mm'], alpha=0.3, color='purple')

        # Rotate x-axis labels for better readability
        for ax in axes:
            ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()

        # Print statistics
        print("\n=== Data Statistics ===")
        print("\nColor Sensor (TCS3200):")
        print(f"  Red   - Min: {df['Red'].min()}, Max: {df['Red'].max()}, Avg: {df['Red'].mean():.2f}")
        print(f"  Green - Min: {df['Green'].min()}, Max: {df['Green'].max()}, Avg: {df['Green'].mean():.2f}")
        print(f"  Blue  - Min: {df['Blue'].min()}, Max: {df['Blue'].max()}, Avg: {df['Blue'].mean():.2f}")
        print(f"\nDistance Sensor (VL53L0X):")
        print(f"  Min: {df['Distance_mm'].min()} mm")
        print(f"  Max: {df['Distance_mm'].max()} mm")
        print(f"  Avg: {df['Distance_mm'].mean():.2f} mm")

    except FileNotFoundError:
        print(f"Error: Could not find '{CSV_FILENAME}'")
        print("Make sure you've run the data logging script first.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    plot_sensor_data()