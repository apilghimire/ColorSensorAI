import pandas as pd
import os
from datetime import datetime

# ======= CONFIGURATION =======
# Define your CSV files and their corresponding labels
# Format: 'filename.csv': 'Label'
CSV_FILES = {
    'Red.csv': 'Red',
    'Blue.csv': 'Blue',
    'Green.csv': 'Green',
    #'Yellow.csv': 'Yellow',
    #'White.csv': 'White',
    #'Black.csv': 'Black',
    # Add more files as needed
}

OUTPUT_FILE = 'Data.csv'

def merge_csv_files():
    """Merge multiple CSV files with labels into a single dataset"""
    
    all_data = []
    total_samples = 0
    
    print("="*70)
    print("MERGING CSV FILES")
    print("="*70)
    
    # Process each CSV file
    for filename, label in CSV_FILES.items():
        if not os.path.exists(filename):
            print(f"⚠ Warning: File '{filename}' not found, skipping...")
            continue
        
        try:
            # Read CSV file
            df = pd.read_csv(filename)
            
            # Select only the required columns (R, G, B, C, Distance)
            # Handle both 'Distance_mm' and 'Distance' column names
            distance_col = 'Distance_mm' if 'Distance_mm' in df.columns else 'Distance'
            
            df_selected = df[['Red', 'Green', 'Blue', 'Clear', distance_col]].copy()
            
            # Rename distance column to standard name
            df_selected.rename(columns={distance_col: 'Distance'}, inplace=True)
            
            # Add label column
            df_selected['Label'] = label
            
            # Append to list
            all_data.append(df_selected)
            
            samples = len(df_selected)
            total_samples += samples
            print(f"✓ Loaded '{filename}': {samples} samples → Label: '{label}'")
            
        except Exception as e:
            print(f"✗ Error loading '{filename}': {e}")
            continue
    
    if not all_data:
        print("\n✗ No valid CSV files found!")
        print("\nAvailable CSV files in current directory:")
        for file in os.listdir('.'):
            if file.endswith('.csv'):
                print(f"  - {file}")
        return
    
    # Concatenate all dataframes
    merged_df = pd.concat(all_data, ignore_index=True)
    
    # Reorder columns: R, G, B, C, Distance, Label
    merged_df = merged_df[['Red', 'Green', 'Blue', 'Clear', 'Distance', 'Label']]
    
    # Save to CSV
    merged_df.to_csv(OUTPUT_FILE, index=False)
    
    print("\n" + "="*70)
    print("MERGE COMPLETE")
    print("="*70)
    print(f"Total samples: {total_samples}")
    print(f"Output file: {OUTPUT_FILE}")
    print("\nSamples per label:")
    print(merged_df['Label'].value_counts().sort_index())
    print("="*70)

def auto_detect_and_merge():
    """Automatically detect CSV files and extract labels from filenames"""
    
    print("="*70)
    print("AUTO-DETECTING CSV FILES")
    print("="*70)
    
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv') and f != OUTPUT_FILE]
    
    if not csv_files:
        print("No CSV files found in current directory!")
        return
    
    print(f"Found {len(csv_files)} CSV file(s):\n")
    
    detected_files = {}
    
    for i, filename in enumerate(csv_files, 1):
        print(f"{i}. {filename}")
        
        # Try to extract label from filename
        # Example: "sensor_data_red.csv" → "Red"
        # or "red_sample.csv" → "Red"
        name_parts = filename.replace('.csv', '').split('_')
        
        # Look for color keywords
        color_keywords = ['red', 'green', 'blue', 'yellow', 'white', 'black', 
                         'orange', 'purple', 'pink', 'brown', 'gray', 'cyan', 'magenta']
        
        label = None
        for part in name_parts:
            if part.lower() in color_keywords:
                label = part.capitalize()
                break
        
        if label is None:
            label = input(f"   Enter label for '{filename}': ").strip()
        else:
            print(f"   Auto-detected label: '{label}'")
            confirm = input(f"   Use this label? (y/n, default=y): ").strip().lower()
            if confirm == 'n':
                label = input(f"   Enter label for '{filename}': ").strip()
        
        if label:
            detected_files[filename] = label
        print()
    
    if not detected_files:
        print("No files selected for merging!")
        return
    
    # Update global CSV_FILES and merge
    global CSV_FILES
    CSV_FILES = detected_files
    merge_csv_files()

def main():
    """Main function with menu"""
    
    print("\n" + "="*70)
    print("CSV MERGER TOOL")
    print("="*70)
    print("\nChoose an option:")
    print("1. Use predefined file list (edit CSV_FILES in code)")
    print("2. Auto-detect files from current directory")
    print("3. Exit")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == '1':
        merge_csv_files()
    elif choice == '2':
        auto_detect_and_merge()
    elif choice == '3':
        print("Exiting...")
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()