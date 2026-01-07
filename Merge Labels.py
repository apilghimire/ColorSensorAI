import pandas as pd
import os
import glob


def merge_sensor_data():
    """
    Merges multiple CSV files into one data.csv file.
    Each CSV filename (without .csv extension) becomes the label/target.

    Example:
    - apple.csv -> label: "apple"
    - banana.csv -> label: "banana"
    - orange.csv -> label: "orange"

    Output: data.csv with columns: Red, Green, Blue, Distance_mm, Label
    """

    # Get all CSV files in current directory
    csv_files = glob.glob('*.csv')

    # Exclude data.csv itself if it exists
    csv_files = [f for f in csv_files if f != 'data.csv']

    if not csv_files:
        print("No CSV files found in the current directory!")
        print("Please ensure you have CSV files with sensor data.")
        return

    print(f"Found {len(csv_files)} CSV file(s):")
    for f in csv_files:
        print(f"  - {f}")

    all_data = []

    # Process each CSV file
    for csv_file in csv_files:
        try:
            # Extract label from filename (remove .csv extension)
            label = os.path.splitext(csv_file)[0]

            # Read CSV file
            df = pd.read_csv(csv_file)

            # Check if required columns exist
            required_cols = ['Red', 'Green', 'Blue', 'Distance_mm']
            if not all(col in df.columns for col in required_cols):
                print(f"Warning: {csv_file} missing required columns. Skipping...")
                continue

            # Add label column
            df['Label'] = label

            # Select only the required columns
            df_selected = df[['Red', 'Green', 'Blue', 'Distance_mm', 'Label']]

            all_data.append(df_selected)

            print(f"✓ Loaded {len(df_selected)} samples from {csv_file} (Label: '{label}')")

        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            continue

    if not all_data:
        print("\nNo valid data found to merge!")
        return

    # Combine all dataframes
    merged_df = pd.concat(all_data, ignore_index=True)

    # Save to data.csv
    merged_df.to_csv('data.csv', index=False)

    print(f"\n{'=' * 50}")
    print(f"✓ Successfully created data.csv")
    print(f"{'=' * 50}")
    print(f"Total samples: {len(merged_df)}")
    print(f"\nLabel distribution:")
    print(merged_df['Label'].value_counts())
    print(f"\nFirst few rows:")
    print(merged_df.head(10))
    print(f"\nFile saved as: data.csv")


if __name__ == "__main__":
    merge_sensor_data()