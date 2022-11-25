ray  stop -f || true
TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=34359738368 ray start --address=globaltpu2.aakashserver.org --resources="{\"tpu\": 1}"