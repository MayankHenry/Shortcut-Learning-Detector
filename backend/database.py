from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
import datetime

# This creates a local file named 'predictions.db' in your backend folder
SQLALCHEMY_DATABASE_URL = "sqlite:///./predictions.db"

# Connect to the SQLite database
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Define the exact table structure for our logs
class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    model_type = Column(String, index=True) # 'biased' or 'unbiased'
    predicted_class = Column(String)        # e.g., 'Digit 1'
    confidence = Column(Float)              # e.g., 98.5