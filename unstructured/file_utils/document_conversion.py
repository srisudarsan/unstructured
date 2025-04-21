from typing import List, Optional
from langchain_core.documents import Document
from unstructured.documents.elements import Element


class TextElement(Element):
    def __init__(self, text: str, metadata: Optional[dict] = None):
        super().__init__(metadata=metadata)
        self.text = text
        self.category = "Text"


def documents_to_elements(documents: List[Document]) -> List[Element]:
    """Convert a list of LangChain Document objects to a list of Elements."""
    elements = []
    for doc in documents:
        element = TextElement(text=doc.page_content, metadata=doc.metadata)
        elements.append(element)
    return elements
