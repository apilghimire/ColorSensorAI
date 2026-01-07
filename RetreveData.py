import serial
import csv
import time
from datetime import datetime

# Configuration
SERIAL_PORT = '/dev/cu.wchusbserial14140'  # Change this to your Arduino port (e.g., 'COM3' on Windows, '/dev/ttyUSB0' on Linux, '/dev/cu.usbmodem14101' on Mac)
BAUD_RATE = 9600
CSV_FILENAME = 'White.csv'


def main():
    print(f"Connecting to {SERIAL_PORT}...")

    try:
        # Open serial connection
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)  # Wait for Arduino to reset

        print(f"Connected! Logging data to {CSV_FILENAME}")
        print("Press Ctrl+C to stop logging\n")

        # Create/open CSV file
        with open(CSV_FILENAME, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)

            # Write header
            csv_writer.writerow(['Timestamp', 'Red', 'Green', 'Blue', 'Distance_mm'])

            # Skip initialization messages
            while True:
                line = ser.readline().decode('utf-8').strip()
                if line and not line.startswith('Initializing') and not line.startswith('Sensors'):
                    break

            # Read and log data
            while True:
                try:
                    # Read line from serial
                    line = ser.readline().decode('utf-8').strip()

                    if line:
                        # Split comma-separated values
                        values = line.split(',')

                        if len(values) == 4:
                            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                            red, green, blue, distance = values

                            # Write to CSV
                            csv_writer.writerow([timestamp, red, green, blue, distance])
                            csvfile.flush()  # Ensure data is written immediately

                            # Print to console
                            print(f"{timestamp} | R:{red} G:{green} B:{blue} Dist:{distance}mm")

                except KeyboardInterrupt:
                    print("\n\nStopping data logging...")
                    break
                except Exception as e:
                    print(f"Error: {e}")
                    continue

        ser.close()
        print(f"Data saved to {CSV_FILENAME}")

    except serial.SerialException as e:
        print(f"Error: Could not open serial port {SERIAL_PORT}")
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