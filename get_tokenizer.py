import torch
from utils import TokenizerType, UnknownTokenizerError
from transformers import AutoTokenizer
import tiktoken


class MyTokenizer:
    def __init__(self, tokenizer_name):
        super().__init__()
        self.tokenizer_name = tokenizer_name
        self.enc_tt = tiktoken.get_encoding("r50k_base")
        self.enc_wp = AutoTokenizer.from_pretrained("bert-base-cased")

    def _tokenize_with_tiktoken(self, text):
        vocab_size = self.enc_tt.n_vocab
        data = torch.tensor(self.enc_tt.encode(text), dtype=torch.long)

        return vocab_size, data

    def _tokenize_with_tfree(self, text):
        # https://github.com/Aleph-Alpha/trigrams
        pass

    def _tokenize_with_wordpiece(self, text):
        # FIXME: Model can't handle the whole sequence at once, loop It
        data = torch.tensor(self.enc_wp.convert_tokens_to_ids(self.enc_wp.tokenize(text)), dtype=torch.long)

        vocab_size = self.enc_wp.vocab_size

        return vocab_size, data

    def encode(self, text):
        if self.tokenizer_name == TokenizerType.TIKTOKEN:
            vocab_size, data = self._tokenize_with_tiktoken(text)
        elif self.tokenizer_name == TokenizerType.TFREE:
            pass
        elif self.tokenizer_name == TokenizerType.WORDPIECE:
            vocab_size, data = self._tokenize_with_wordpiece(text)
        else:
            raise UnknownTokenizerError(self.tokenizer_name)

        return vocab_size, data

    def decode(self, context):
        if self.tokenizer_name == TokenizerType.TIKTOKEN:
            results = self.enc_tt.decode(context)
        elif self.tokenizer_name == TokenizerType.TFREE:
            pass
        elif self.tokenizer_name == TokenizerType.WORDPIECE:
            results = self.enc_wp.decode(context)
        else:
            raise UnknownTokenizerError(self.tokenizer_name)

        return results
