"""
This script controls the logging functionality. When the classifer detects a dog bark, this script will log the event to a csv file.
This script also generates a separate csv file for each day.
"""

import pandas as pd
import os
from datetime import datetime

class Logger:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        self.create_log_dir()


    def create_log_dir(self):
        """Create the log directory if it doesn't exist"""
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            print(f"Created log directory: {self.log_dir}")

    def log_event(self, event_type, event_data):
        """Log an event to the log file"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Create a new log file for the current day if it doesn't exist
        today = datetime.now().strftime("%Y-%m-%d")
        log_file = os.path.join(self.log_dir, f"{today}.csv")

        if not os.path.exists(log_file):
            with open(log_file, "w") as f:
                f.write("timestamp,event_type,event_data\n")

        # Append the event to the log file
        # TODO: Add a check to see if the event is already in the file.
        # If it is, don't log it again.
        # TODO: Add a check to see if the event is a duplicate.
        with open(log_file, "a") as f:
            f.write(f"{timestamp},{event_type},{event_data}\n")

        print(f"Logged event: {event_type} at {timestamp}")

    def get_log_file_path(self):
        """Get the path to the log file for the current day"""
        today = datetime.now().strftime("%Y-%m-%d")
        return os.path.join(self.log_dir, f"{today}.csv")