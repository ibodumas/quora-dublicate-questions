import os
import sys


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
_path = [
            os.path.join(ROOT_DIR, "src"),
            os.path.join(ROOT_DIR, "src", "data"),
            os.path.join(ROOT_DIR, "src", "api"),
            os.path.join(ROOT_DIR, "src", "evaluate"),
            os.path.join(ROOT_DIR, "src", "model"),
            os.path.join(ROOT_DIR, "src", "quoradata")
        ]

for pth in _path:
    if pth not in sys.path:
        sys.path.append(pth)


