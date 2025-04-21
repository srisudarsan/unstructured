import json
from threading import Lock
import os

from transformers import AutoTokenizer
from unstructured.documents.elements import Element
from unstructured.file_utils.document_conversion import documents_to_elements

from typing import Iterable, List, Tuple, Any, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

DEFAULT_PATH = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
    "vendor_packages",
    "{}"
)


class Tokenizer:
    """
    Singleton class to load tokenizer for a given model
    """

    _instances = {}
    _lock = Lock()

    def __new__(cls, model_name: str):
        with cls._lock:
            if model_name not in cls._instances:  # Only load if not already cached
                instance = super().__new__(cls)
                instance.model_tokenizer = AutoTokenizer.from_pretrained(
                    DEFAULT_PATH.format(model_name)
                )
                cls._instances[model_name] = instance
        return cls._instances[model_name]

    def get_tokenizer(self):
        return self.model_tokenizer


def get_tokenizer(model_name: str):
    try:
        import os

        print(os.listdir(DEFAULT_PATH.format(model_name)))
        return Tokenizer(model_name).get_tokenizer()
    except Exception as e:
        raise e


def _get_char_splitter(
        chunk_max_characters: int,
        chunk_overlap: int
) -> RecursiveCharacterTextSplitter:
    """
    Creates a RecursiveCharacterTextSplitter instance for splitting documents into
    character-based chunks with specified size and overlap.

    This splitter attempts to split text using a prioritized list of separators, such as
    newlines, spaces, and various punctuation marks—including support for fullwidth and
    ideographic characters often used in multilingual text.

    Parameters
    ----------
    chunk_max_characters : int
        The maximum number of characters per chunk.
    chunk_overlap : int
        The number of characters to overlap between chunks for context preservation.

    Returns
    -------
    RecursiveCharacterTextSplitter
        An instance of RecursiveCharacterTextSplitter configured with the given parameters.
    """
    return RecursiveCharacterTextSplitter(
        separators=[
            "\n\n",
            "\n",
            " ",
            ".",
            ",",
            "\u200b",  # Zero-width space
            "\uff0c",  # Fullwidth comma
            "\u3001",  # Ideographic comma
            "\uff0e",  # Fullwidth full stop
            "\u3002",  # Ideographic full stop
            "",
        ],
        chunk_size=chunk_max_characters,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )


def recursive_text_splitter(
        elements: Iterable[Element],
        model_name: str,
        chunk_max_characters: int,
        chunk_overlap: int,
        metadata,
) -> list[Element]:
    """
    Splits the document content into smaller chunks using a tokenizer-aware strategy.

    Parameters
    ----------
    model_name : str
        The name of the model whose tokenizer will be used for token-aware splitting.
    chunk_max_characters : int
        The maximum number of characters (or tokens) per chunk.
    chunk_overlap : int
        The number of overlapping characters (or tokens) between chunks.

    Returns
    -------
    list[Element]
        A list of document chunks with associated metadata.
    """
    metadata_dict = {}
    if metadata and isinstance(metadata, str):
        try:
            metadata_dict = json.loads(metadata)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON string passed for metadata.")

    try:
        tokenizer = get_tokenizer(model_name)
        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer=tokenizer,
            chunk_size=chunk_max_characters,
            chunk_overlap=chunk_overlap,
        )
        text_splitter._length_function = lambda x: len(tokenizer.tokenize(x))
    except Exception:
        text_splitter = _get_char_splitter(chunk_max_characters, chunk_overlap)

    # Extract texts from elements (fed search ingestion logic)
    concatenated_text = " ".join(
        [element.text for element in elements]
    )

    document_list: list[Document] = text_splitter.create_documents([concatenated_text], [])

    return documents_to_elements(document_list)


def custom_chunking(
        elements: Iterable[Element],
        *,
        max_characters: int = 256,
        new_after_n_chars: int = 128,
        chunking_model_name: Optional[str] = None,
        custom_metadata: Optional[str] = None,
) -> list[Element]:
    """
    Custom chunking function for processing and splitting text elements into smaller chunks.
    This function requires a valid `model_name`, `chunk_size`, and `chunk_overlap` in `kwargs`.

    Args:
        elements (Iterable[Element]): A sequence of Element objects containing text to be chunked.
        **kwargs (Any): Additional parameters including 'model_name', 'chunk_size', and 'chunk_overlap'.

    Returns:
        list[Element]: A list of chunked Element objects after processing the text.

    Raises:
        ValueError: If 'model_name' is not provided in `kwargs`.

    Parameters
    ----------
    max_characters
    new_after_n_chars
    custom_metadata
    elements
    chunking_model_name
    """

    model_name = chunking_model_name
    chunk_size = max_characters
    chunk_overlap = new_after_n_chars

    # Validating if a model_name is provided
    if not model_name:
        raise ValueError(
            f"A 'model_name' must be provided when using the 'custom' chunking strategy. provided {model_name}")

    return recursive_text_splitter(elements, model_name, chunk_size, chunk_overlap, custom_metadata)


from unstructured.documents.elements import Element, ElementMetadata


class SimpleElement(Element):
    def __init__(self, text: str, metadata: Optional[dict] = None):
        super().__init__(metadata=ElementMetadata.from_dict(metadata or {}))
        self.text = text


if __name__ == "__main__":
    element_dicts = [
        {'type': 'CompositeElement', 'element_id': '9c10ac904cf91c94380118a74464b8e8', 'text': 'Page 1',
         'metadata': {'languages': ['eng'], 'page_number': 1, 'orig_elements': '...base64...',
                      'filename': 'large_random_content.pdf', 'filetype': 'application/pdf'}},
        {'type': 'CompositeElement', 'element_id': '11a50a2148241f6e77811bf1347d4cfd',
         'text': 'Feel them suddenly own. Page visit space at much...',
         'metadata': {'languages': ['eng'], 'page_number': 1, 'orig_elements': '...base64...',
                      'filename': 'large_random_content.pdf', 'filetype': 'application/pdf'}}
    ]

    # Convert dicts to SimpleElement instances
    elements = []
    for elem in element_dicts:
        metadata = elem.get("metadata", {})
        metadata.pop("orig_elements", None)  # remove invalid base64 field
        elements.append(SimpleElement(text=elem["text"], metadata=metadata))

    # Define chunking parameters
    kwargs = {
        "model_name": "BAAI-bge-m3",
        "chunk_size": 256,
        "chunk_overlap": 128,
        "metadata": elements[0].metadata.to_dict(),  # sample metadata for testing
    }

    # Call custom chunking
    chunks = custom_chunking(elements)

    # Print result
    for chunk in chunks:
        print(chunk)
