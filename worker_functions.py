import time

def cpu_intensive_work(n):
    """Heavy computation"""
    total = 0
    for i in range(n * 1_000_000):
        total += i ** 2
    return total

def io_intensive_work(delay):
    """I/O simulation"""
    time.sleep(delay)
    return f"Slept for {delay} seconds"

