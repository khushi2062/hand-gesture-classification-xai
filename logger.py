import sys
import os
from datetime import datetime

class Tee:
    def __init__(self, filepath):
        self.file = open(filepath, 'a', encoding='utf-8')
        self.terminal = sys.stdout

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)

    def flush(self):
        self.terminal.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def start_logger(phase_name):
    os.makedirs('../results', exist_ok=True)

    # single fixed filename — all phases append to same file
    filepath = '../results/project_results.txt'

    sys.stdout = Tee(filepath)

    # header without time — just phase name
    print("=" * 60)
    print(f"Phase: {phase_name}")
    print("=" * 60)
    print()

    return filepath