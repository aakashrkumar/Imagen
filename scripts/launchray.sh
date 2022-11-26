ray  stop -f || true
TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=34359738368 ray start --head --port=6379 --node-ip-address=globaltpu2.aakashserver.org --resources="{\"tpu\": 1, \"host\":999}" --dashboard-host=0.0.0.0
