from datetime import datetime

def print_log(message):
    """Timestamped log utility for pipeline progress tracking."""
    time = datetime.now().strftime("%H:%M:%S")
    print(f"\n[{time}] {message}")
