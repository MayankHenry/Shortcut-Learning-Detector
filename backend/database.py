from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
import datetime

# Local SQLite Database
SQLALCHEMY_DATABASE_URL = "sqlite:///./predictions.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    model_type = Column(String, index=True)
    predicted_class = Column(String)
    confidence = Column(Float)
    
    # --- NEW: Cloud Object Storage URLs ---
    original_image_url = Column(String, nullable=True) 
    heatmap_url = Column(String, nullable=True)