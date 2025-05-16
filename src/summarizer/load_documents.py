from datasets import load_dataset
import random
from pydantic import BaseModel
from typing import List, Dict, Tuple


class Question(BaseModel):
    q: str
    a: str


class Document(BaseModel):
    document_text: str
    questions: List[Question]


def load_documents() -> Tuple[List[Document], List[Document]]:
    ds = load_dataset("ServiceNow/repliqa")
    documents: Dict[str, Document] = {}

    for data in ds["repliqa_0"]:
        if data["document_id"] not in documents:
            documents[data["document_id"]] = Document(
                document_text=data["document_extracted"],
                questions=[],
            )
        documents[data["document_id"]].questions.append(
            Question(q=data["question"], a=data["answer"])
        )

    all_documents: List[Document] = []
    for doc_id in documents:
        all_documents.append(documents[doc_id])

    random.seed(80)
    random.shuffle(all_documents)

    # Now split into train/val/test sets
    val_size = min(25, len(all_documents) // 10)

    val_documents = all_documents[:val_size]
    train_documents = all_documents[val_size:]

    return val_documents, train_documents
