import os

# --- Redis Configuration ---
# Defaults to 'redis' for Docker Compose networking, 'localhost' for local dev
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

# --- Stream and Channel Names ---
INGEST_STREAM_NAME = "pdm:ingest_stream"
RESULT_CHANNEL_NAME = "pdm:result_channel"

# --- Worker Consumer Group Name ---
CONSUMER_GROUP_NAME = "pdm_worker_group"

# --- Security and JWT Configuration ---
# For production, this MUST be set as an environment variable.
# Generate a good secret with: openssl rand -hex 32
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "a-secure-default-secret-for-development")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# --- Alerting Configuration ---
ALERT_THRESHOLD = 15.0  # Anomaly score above which we consider a machine "in alert"
ALERT_DURATION_SECONDS = 120 # How long the score must be above threshold to trigger a formal notification

# --- SMTP Configuration for Email Alerts ---
# For production, these MUST be environment variables, ideally managed by a secrets manager.
SMTP_ENABLED = os.getenv("SMTP_ENABLED", "true").lower() in ("true", "1", "t")
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SMTP_USER = os.getenv("SMTP_USER", "gafenatesting@gmail.com") # Should be your sending email
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "clzq axex gyzv cgwe") # Your email App Password
SMTP_SENDER_EMAIL = os.getenv("SMTP_SENDER_EMAIL", "pdm-system@example.com") # "From" address

# NOTE: SMTP_RECIPIENT_EMAIL has been removed.
# The list of recipients is now fetched dynamically from the user database.