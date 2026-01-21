import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os

# ======= CONFIGURATION =======
CSV_FILE = 'unspecified.csv'  # Change to your CSV filename

def load_and_prepare_data(filename):
    """Load CSV data and prepare for analysis"""
    try:
        df = pd.read_csv(filename)
        
        # Convert timestamp to datetime if present
        if 'Timestamp' in df.columns:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            # Calculate time elapsed in seconds from start
            df['Time_sec'] = (df['Timestamp'] - df['Timestamp'].iloc[0]).dt.total_seconds()
        else:
            # If no timestamp, create sample index
            df['Time_sec'] = np.arange(len(df)) * 0.2  # Assuming 200ms between samples
        
        return df
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found!")
        print(f"Current directory: {os.getcwd()}")
        print("Available CSV files:")
        for file in os.listdir('.'):
            if file.endswith('.csv'):
                print(f"  - {file}")
        return None

def calculate_statistics(df):
    """Calculate and print statistics for each sensor"""
    print("\n" + "="*70)
    print("SENSOR DATA STATISTICS")
    print("="*70)
    
    sensors = ['Red', 'Green', 'Blue', 'Clear', 'Distance_mm']
    
    for sensor in sensors:
        if sensor in df.columns:
            print(f"\n{sensor}:")
            print(f"  Mean:   {df[sensor].mean():.2f}")
            print(f"  Median: {df[sensor].median():.2f}")
            print(f"  Std:    {df[sensor].std():.2f}")
            print(f"  Min:    {df[sensor].min():.2f}")
            print(f"  Max:    {df[sensor].max():.2f}")
    
    print(f"\nTotal samples: {len(df)}")
    if 'Time_sec' in df.columns:
        duration = df['Time_sec'].iloc[-1]
        print(f"Duration: {duration:.2f} seconds")
        print(f"Average sample rate: {len(df)/duration:.2f} samples/sec")
    print("="*70 + "\n")

def plot_all_data(df):
    """Create comprehensive visualization of all sensor data"""
    
    fig = plt.figure(figsize=(16, 10))
    
    # Plot 1: RGB Values over time
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(df['Time_sec'], df['Red'], 'r-', label='Red', linewidth=1.5, alpha=0.7)
    ax1.plot(df['Time_sec'], df['Green'], 'g-', label='Green', linewidth=1.5, alpha=0.7)
    ax1.plot(df['Time_sec'], df['Blue'], 'b-', label='Blue', linewidth=1.5, alpha=0.7)
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('RGB Value')
    ax1.set_title('RGB Color Sensor Data Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Clear Channel
    ax2 = plt.subplot(3, 2, 2)
    ax2.plot(df['Time_sec'], df['Clear'], 'k-', linewidth=1.5)
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Clear Channel Value')
    ax2.set_title('Clear Channel (Luminosity) Over Time')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Distance
    ax3 = plt.subplot(3, 2, 3)
    # Filter out invalid readings (-1)
    valid_distance = df[df['Distance_mm'] >= 0]
    ax3.plot(valid_distance['Time_sec'], valid_distance['Distance_mm'], 
             'purple', linewidth=1.5)
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Distance (mm)')
    ax3.set_title('VL53L0X Distance Measurements')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: RGB Ratios (normalized)
    ax4 = plt.subplot(3, 2, 4)
    total_rgb = df['Red'] + df['Green'] + df['Blue']
    total_rgb = total_rgb.replace(0, 1)  # Avoid division by zero
    ax4.plot(df['Time_sec'], df['Red']/total_rgb, 'r-', label='Red %', alpha=0.7)
    ax4.plot(df['Time_sec'], df['Green']/total_rgb, 'g-', label='Green %', alpha=0.7)
    ax4.plot(df['Time_sec'], df['Blue']/total_rgb, 'b-', label='Blue %', alpha=0.7)
    ax4.set_xlabel('Time (seconds)')
    ax4.set_ylabel('Normalized Ratio')
    ax4.set_title('RGB Color Ratios (Normalized)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Histogram of Distance
    ax5 = plt.subplot(3, 2, 5)
    valid_dist_values = df[df['Distance_mm'] >= 0]['Distance_mm']
    ax5.hist(valid_dist_values, bins=50, color='purple', alpha=0.7, edgecolor='black')
    ax5.set_xlabel('Distance (mm)')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Distance Distribution')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: RGB in 3D color space preview
    ax6 = plt.subplot(3, 2, 6)
    # Sample every Nth point for clarity
    step = max(1, len(df) // 100)
    scatter_data = df.iloc[::step]
    
    # Normalize RGB for color display
    max_rgb = scatter_data[['Red', 'Green', 'Blue']].max().max()
    colors = scatter_data[['Red', 'Green', 'Blue']].values / max_rgb
    colors = np.clip(colors, 0, 1)  # Ensure values are in [0,1]
    
    scatter = ax6.scatter(scatter_data['Red'], scatter_data['Green'], 
                         c=colors, s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax6.set_xlabel('Red Value')
    ax6.set_ylabel('Green Value')
    ax6.set_title('RGB Color Space (Red vs Green)')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_filename = CSV_FILE.replace('.csv', '_analysis.png')
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {output_filename}")
    
    plt.show()

def plot_correlation_analysis(df):
    """Plot correlation between distance and color readings"""
    
    valid_data = df[df['Distance_mm'] >= 0].copy()
    
    if len(valid_data) == 0:
        print("No valid distance data for correlation analysis")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Red vs Distance
    axes[0, 0].scatter(valid_data['Distance_mm'], valid_data['Red'], 
                       c='red', alpha=0.5, s=20)
    axes[0, 0].set_xlabel('Distance (mm)')
    axes[0, 0].set_ylabel('Red Value')
    axes[0, 0].set_title('Red vs Distance')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Green vs Distance
    axes[0, 1].scatter(valid_data['Distance_mm'], valid_data['Green'], 
                       c='green', alpha=0.5, s=20)
    axes[0, 1].set_xlabel('Distance (mm)')
    axes[0, 1].set_ylabel('Green Value')
    axes[0, 1].set_title('Green vs Distance')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Blue vs Distance
    axes[1, 0].scatter(valid_data['Distance_mm'], valid_data['Blue'], 
                       c='blue', alpha=0.5, s=20)
    axes[1, 0].set_xlabel('Distance (mm)')
    axes[1, 0].set_ylabel('Blue Value')
    axes[1, 0].set_title('Blue vs Distance')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Clear vs Distance
    axes[1, 1].scatter(valid_data['Distance_mm'], valid_data['Clear'], 
                       c='black', alpha=0.5, s=20)
    axes[1, 1].set_xlabel('Distance (mm)')
    axes[1, 1].set_ylabel('Clear Value')
    axes[1, 1].set_title('Clear vs Distance')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_filename = CSV_FILE.replace('.csv', '_correlation.png')
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Correlation plot saved as: {output_filename}")
    
    plt.show()

def main():
    """Main analysis function"""
    print(f"Loading data from: {CSV_FILE}")
    
    df = load_and_prepare_data(CSV_FILE)
    
    if df is None:
        return
    
    print(f"Successfully loaded {len(df)} samples")
    
    # Calculate and print statistics
    calculate_statistics(df)
    
    # Create visualizations
    print("Generating plots...")
    plot_all_data(df)
    plot_correlation_analysis(df)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()