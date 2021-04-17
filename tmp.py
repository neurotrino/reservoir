import subprocess

s = subprocess.check_output(
    ["git", "status", "--porcelain"]
).strip().decode('utf-8')

print(s)

if s is None:
    print('None')
