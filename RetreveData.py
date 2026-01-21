import serial
import csv
import time
from datetime import datetime

# ======= CONFIGURATION =======
SERIAL_PORT = '/dev/cu.wchusbserial58FA0447301'  # Change to your port (COM3, COM4 on Windows; /dev/ttyUSB0, /dev/ttyACM0 on Linux/Mac)
BAUD_RATE = 115200
CSV_FILENAME = 'unspecified.csv'
MAX_SAMPLES = 600  # Set to desired number of samples


# Column headers matching your Arduino output
HEADERS = ['Red', 'Green', 'Blue', 'Clear', 'Distance_mm', 'Timestamp']

def read_serial_data():
    """Read data from ESP32-S3 and save to CSV file"""
    
    sample_count = 0
    
    try:
        # Open serial connection
        print(f"Connecting to {SERIAL_PORT} at {BAUD_RATE} baud...")
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)  # Wait for connection to stabilize
        
        # Clear any initial garbage data
        ser.reset_input_buffer()
        print("Connected! Reading data...\n")
        
        # Open CSV file for writing
        with open(CSV_FILENAME, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(HEADERS)  # Write header row
            
            print(f"Saving data to: {CSV_FILENAME}")
            print(f"Target samples: {MAX_SAMPLES}")
            print("Press Ctrl+C to stop early\n")
            
            while sample_count < MAX_SAMPLES:
                try:
                    # Read line from serial port
                    if ser.in_waiting > 0:
                        line = ser.readline().decode('utf-8').strip()
                        
                        # Skip empty lines or error messages
                        if not line or 'NOT FOUND' in line:
                            continue
                        
                        # Parse CSV data
                        data = line.split(',')
                        
                        # Validate data format (should have 5 values: R,G,B,C,Distance)
                        if len(data) == 5:
                            sample_count += 1
                            
                            # Add timestamp
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                            data.append(timestamp)
                            
                            # Write to CSV
                            writer.writerow(data)
                            csvfile.flush()  # Ensure data is written immediately
                            
                            # Print only sample number
                            print(f"Sample: {sample_count}/{MAX_SAMPLES}")
                        
                except UnicodeDecodeError:
                    # Skip lines with encoding errors
                    continue
            
            # Reached target samples
            print(f"\n✓ Successfully collected {sample_count} samples!")
            print(f"Data saved to: {CSV_FILENAME}")
                    
    except serial.SerialException as e:
        print(f"\nSerial Error: {e}")
        print("Please check:")
        print("1. Correct port name")
        print("2. ESP32 is connected")
        print("3. No other program is using the port")
        
    except KeyboardInterrupt:
        print(f"\n\nStopped by user at {sample_count} samples")
        print(f"Data saved to: {CSV_FILENAME}")
        
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("Serial port closed")

if __name__ == "__main__":
    read_serial_data()