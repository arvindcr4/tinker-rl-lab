import os
import glob

# Get all files of interest (markdown, tex, text)
files = []
for root, dirs, filenames in os.walk('.'):
    if '.git' in root or '__pycache__' in root or '.venv' in root:
        continue
    for f in filenames:
        if f.endswith('.md') or f.endswith('.tex') or f.endswith('.txt'):
            files.append(os.path.join(root, f))

# Sort to make it deterministic
files.sort()

# Divide into 45 chunks
num_chunks = 45
chunks = [[] for _ in range(num_chunks)]
for i, f in enumerate(files):
    chunks[i % num_chunks].append(f)

# Write out the chunks
for i in range(num_chunks):
    with open(f"agent_chunk_{i}.txt", "w") as f:
        for filepath in chunks[i]:
            f.write(filepath + "\n")

print(f"Created 45 chunks. Total files: {len(files)}")
