import subprocess

s = subprocess.check_output(
    ["git", "status", "--porcelain"]
).strip().decode('utf-8')

print(f"-{s}-")

if len(s) == 0:
    print('None')
