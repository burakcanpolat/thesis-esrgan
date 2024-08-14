import os

# List of specific virtual environment directories
env_dirs = ['esrgan_env/']

# Path to the .gitignore file
gitignore_path = '.gitignore'

# Read the current content of .gitignore
if os.path.exists(gitignore_path):
    with open(gitignore_path, 'r') as file:
        gitignore_content = file.read()
else:
    gitignore_content = ""

# Add new environment directories to .gitignore if not already present
with open(gitignore_path, 'a') as file:
    for env_dir in env_dirs:
        if env_dir not in gitignore_content:
            file.write(f"{env_dir}\n")

print(f"Updated {gitignore_path} with environment directory: {env_dirs[0]}")