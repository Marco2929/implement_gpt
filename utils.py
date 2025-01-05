from enum import Enum

class TokenizerType(Enum):
    TIKTOKEN = "tiktoken"
    TFREE = "tfree"
    WORDPIECE = "wordpiece"
    UNIGRAM = "unigram"

class UnknownTokenizerError(Exception):
    """Custom exception for unknown tokenizer types."""
    def __init__(self, tokenizer_type):
        self.tokenizer_type = tokenizer_type
        super().__init__(f"Unknown tokenizer type: {tokenizer_type}")
