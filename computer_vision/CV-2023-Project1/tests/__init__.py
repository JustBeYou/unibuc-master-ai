import os
import sys

tests_dir = os.path.dirname(__file__)
src_dir = os.path.join(tests_dir, '..', 'src')

sys.path.insert(0, os.path.abspath(os.path.join(src_dir)))
sys.path.insert(0, os.path.abspath(os.path.join(tests_dir)))