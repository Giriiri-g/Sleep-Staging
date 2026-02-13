import subprocess

print("Starting training on HPC...")

which_python = "/home/your_user/sleep_staging/env/bin/python"
main_path = "/home/geethalekshmy/GirishS/train.py"

subprocess.run([which_python, main_path])

print("Training finished.")
