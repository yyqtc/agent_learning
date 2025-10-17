from typing_extensions import TypedDict
from typing import List

class PaperInfo(TypedDict):
    title: str
    authors: List[str]
    abstract: str
    submitted_date: str