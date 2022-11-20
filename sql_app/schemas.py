from pydantic import BaseModel
from typing import Union

class Model_Stage(BaseModel):
    name: str
    version: str
    creation_time: int
    last_updated_time: int
    description: str
    user_id: str
    current_stage: str
    source: str
    run_id: str
    status: str
    status_message: str
    run_link: str

    class Config:
        orm_mode = True


