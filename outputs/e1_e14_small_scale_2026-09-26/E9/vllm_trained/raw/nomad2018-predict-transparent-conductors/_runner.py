import subprocess, os
os.chdir('/content/nomad2018-predict-transparent-conductors')
if os.path.exists('submission.csv'): os.remove('submission.csv')
try:
    p = subprocess.run(['python', 'solution.py'], capture_output=True, text=True, timeout=1200)
    rc, so, se = p.returncode, p.stdout, p.stderr
except subprocess.TimeoutExpired:
    rc, so, se = -9, '', 'TIMEOUT'
print('RC=', rc, 'HAS_SUB=', os.path.exists('submission.csv'))
print('STDOUT_TAIL', so[-1500:])
print('STDERR_TAIL', se[-3000:])
