import os
import pickle
import psutil
import sys


def load_data(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data.get('data'), data.get('labels')


def check_memory_usage(limit_mb):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    current_memory = mem_info.rss / (1024 * 1024)
    print(f"Current memory usage: {current_memory} MB")
    if current_memory > limit_mb:
        print(f"Memory limit exceeded: {current_memory} MB used, which is over the {limit_mb} MB limit.")
        sys.exit(1)
