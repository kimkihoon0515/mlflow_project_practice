from sqlalchemy import Boolean, Column, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from .database import Base


class Model_Stage(Base):

    name = Column(String)
    version = Column(String, primary_key=True, index=True)
    creation_time = Column(Integer)
    last_updated_time = Column(Integer)
    description = Column(String)
    user_id = Column(String)
    current_stage = Column(String)
    source = Column(String)
    run_id = Column(String)
    status = Column(String)
    status_message = Column(String)
    run_link = Column(String)

