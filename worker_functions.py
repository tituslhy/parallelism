import time
import asyncio
from concurrent.futures import ProcessPoolExecutor

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

async def async_cpu_test():
    cpu_tasks = [100, 200, 300, 400]
    loop = asyncio.get_running_loop()
    start = time.time()
    
    with ProcessPoolExecutor() as executor:
        results = await asyncio.gather(*[
            loop.run_in_executor(
                executor,
                cpu_intensive_work,
                task,
            ) for task in cpu_tasks
        ])
    print(f"Asyncio + ProcessPool (CPU): {time.time() - start:.2f}s")