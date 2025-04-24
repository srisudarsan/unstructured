from typing import List, Optional
from langchain_core.documents import Document
from unstructured.documents.elements import Element, ElementMetadata


class TextElement(Element):
    def __init__(self, text: str, metadata: Optional[ElementMetadata] = None):
        super().__init__(metadata=metadata)
        self.text = text
        self.category = "Text"


def documents_to_elements(documents: List[Document]) -> List[Element]:
    """Convert a list of LangChain Document objects to a list of Elements."""
    elements = []
    for doc in documents:
        meta = ElementMetadata()
        if doc.metadata:
            meta.data.update(doc.metadata)
        element = TextElement(text=doc.page_content, metadata=meta)
        elements.append(element)
    return elements
