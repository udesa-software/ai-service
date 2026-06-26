import os

RUN_REAL_EMBEDDINGS_BENCHMARK = "RUN_REAL_EMBEDDINGS_BENCHMARK"

# Set dummy env vars before any module is imported.
# The service modules instantiate Settings() at import time,
# so these must exist before collection starts.
os.environ.setdefault("JWT_SECRET", "test-secret")
os.environ.setdefault("USERS_INTERNAL_URL", "http://users:3000")
os.environ.setdefault("FRIENDS_INTERNAL_URL", "http://friends:3001")
os.environ.setdefault("INTERNAL_SECRET", "test-internal-secret")
os.environ.setdefault("USERS_DB_HOST", "localhost")
os.environ.setdefault("USERS_DB_USER", "test-user")
os.environ.setdefault("USERS_DB_PASSWORD", "test-password")

if os.getenv(RUN_REAL_EMBEDDINGS_BENCHMARK, "").lower() not in {"1", "true", "yes"}:
    os.environ.setdefault("HF_API_TOKEN", "hf_test_token")
