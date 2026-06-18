import os

# Set dummy env vars before any module is imported.
# The service modules instantiate Settings() at import time,
# so these must exist before collection starts.
os.environ.setdefault("JWT_SECRET", "test-secret")
os.environ.setdefault("USERS_INTERNAL_URL", "http://users:3000")
os.environ.setdefault("FRIENDS_INTERNAL_URL", "http://friends:3001")
os.environ.setdefault("INTERNAL_SECRET", "test-internal-secret")
os.environ.setdefault("HF_API_TOKEN", "hf_test_token")
