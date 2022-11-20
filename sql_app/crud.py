from sqlalchemy.orm import Session

from . import models, schemas

def get_model_stage(db: Session, name: str):
    return db.query(models.Model_Stage).filter(models.Model_Stage.name == name).first()

