import os
import sys

# Get the current directory
current_dir = os.path.abspath(os.path.dirname(__file__))

# Move one level up from the current directory
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardiru))
# Add the parent directory to the system path
sys.path.append(parent_dir)