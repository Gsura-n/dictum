"""Runs inside the sandbox: execute one command (stdin) in a fresh fixture copy
and print JSON with stdout, exit code and a filesystem snapshot."""
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile

cmd = sys.stdin.read()
FIXTURE = os.environ.get("DICTUM_FIXTURE", "/fixture")
RUNS = os.environ.get("DICTUM_RUNS", "/runs")
root = tempfile.mkdtemp(prefix="run", dir=RUNS)
subprocess.run(["cp", "-a", "--sparse=always", FIXTURE, f"{root}/fx"], check=True)
home, cwd = f"{root}/fx/home", f"{root}/fx/work"
env = {"HOME": home, "PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8", "USER": "dev",
       "TERM": "dumb", "COLUMNS": "120"}

timed_out = False
p = subprocess.Popen(["bash", "-c", cmd], cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
try:
    out, err = p.communicate(timeout=5)
except subprocess.TimeoutExpired:
    os.killpg(p.pid, signal.SIGKILL)
    out, err = p.communicate()
    timed_out = True

snap = []
for dirpath, dirnames, filenames in os.walk(f"{root}/fx"):
    dirnames.sort()
    for name in sorted(dirnames + filenames):
        path = os.path.join(dirpath, name)
        rel = os.path.relpath(path, f"{root}/fx")
        st = os.lstat(path)
        entry = [rel, oct(st.st_mode & 0o7777)]
        if os.path.islink(path):
            entry += ["link", os.readlink(path)]
        elif os.path.isdir(path):
            entry += ["dir"]
        else:
            entry += ["file", st.st_size]
            if st.st_size <= 5_000_000:
                try:
                    with open(path, "rb") as fh:
                        entry.append(hashlib.sha1(fh.read()).hexdigest())
                except OSError:
                    entry.append("unreadable")
        snap.append(entry)

text = out.decode("utf-8", "replace")[:50000]
for p_ in (cwd + "/", cwd, home + "/", home, f"{root}/fx"):
    text = text.replace(p_, "")
print(json.dumps({"rc": p.returncode, "timed_out": timed_out, "stdout": text,
                  "stderr": err.decode("utf-8", "replace")[:2000],
                  "fs": hashlib.sha1(json.dumps(snap).encode()).hexdigest()}))
subprocess.run(["chmod", "-R", "u+rwx", root], capture_output=True)
subprocess.run(["rm", "-rf", root], capture_output=True)
