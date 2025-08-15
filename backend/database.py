from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

# --- Database URL Configuration ---
# In a real app, you'd use Pydantic settings for this
POSTGRES_USER = os.getenv("POSTGRES_USER", "pdm_user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "pdm_password")
POSTGRES_SERVER = os.getenv("POSTGRES_SERVER", "localhost") # Use 'postgres_db' when running in Docker
POSTGRES_DB = os.getenv("POSTGRES_DB", "pdm_database")

#SQLALCHEMY_DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_SERVER}/{POSTGRES_DB}"
SQLALCHEMY_DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@localhost:5432/{POSTGRES_DB}"
# --- Engine and Session Setup ---
engine = create_engine(SQLALCHEMY_DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# --- Dependency to get a DB session ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()