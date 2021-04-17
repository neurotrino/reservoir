import subprocess

s = subprocess.check_output(
    ["git", "status", "--porcelain"]
).strip().decode('utf-8')

print(f"-{s}-")

if s.isspace():
    print('None')
