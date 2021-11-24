from typing import Dict, Iterable, List

from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import LabelField, TextField, SequenceLabelField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, PretrainedTransformerIndexer, pretrained_transformer_indexer
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer
from transformers import BertTokenizer
from transformers.models import bert


@DatasetReader.register("conll2003-reader")
class ClassificationTsvReader(DatasetReader):
    def __init__(
        self,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        max_tokens: int = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer or WhitespaceTokenizer()
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.max_tokens = max_tokens

    def text_to_instance(self, tokens: List[str], labels: List[str] = None) -> Instance:
        if self.max_tokens:
            tokens = tokens[: self.max_tokens - 2]
            labels = labels[: self.max_tokens - 2]
            tokens = self.tokenizer.add_special_tokens(tokens)
        text_field = TextField(tokens, self.token_indexers)
        fields = {"tokens": text_field}
        # fields = {}
        if labels:
            fields["label"] = SequenceLabelField(labels, text_field, "ner_tags")
        return Instance(fields)

    def _read(self, file_path: str) -> Iterable[Instance]:
        sentence, label = [], []
        with open(file_path, "r") as lines:
            for line in lines:
                if len(line)==0 or line.startswith('-DOCSTART') or line[0]=="\n":
                    if len(sentence) > 0:
                        yield self.text_to_instance(sentence, label)
                    sentence, label = [], []
                    continue
                splits = line.split(' ')
                sentence.append(Token(splits[0]))
                label.append(splits[-1][:-1])
        
        if len(sentence) > 0:
            yield self.text_to_instance(sentence, label)


@DatasetReader.register("bert")
class ClassificationTsvReaderBERT(DatasetReader):
    def __init__(
        self,
        bert_path,
        max_tokens: int = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.max_tokens = max_tokens

    def text_to_instance(self, tokens: List[str], labels: List[str] = None) -> Instance:
        if self.max_tokens:
            tokens = tokens[: self.max_tokens]
            labels = labels[: self.max_tokens]
        tokens = self.tokenizer.add_special_tokens(tokens)
        text_field = TextField(tokens)
        fields = {"tokens": text_field}
        # fields = {}
        if labels:
            fields["label"] = SequenceLabelField(labels, text_field, "ner_tags")
        return Instance(fields)

    def _read(self, file_path: str) -> Iterable[Instance]:
        sentence, label = [], []
        with open(file_path, "r") as lines:
            for line in lines:
                if len(line)==0 or line.startswith('-DOCSTART') or line[0]=="\n":
                    if len(sentence) > 0:
                        yield self.text_to_instance(sentence, label)
                    sentence, label = [], []
                    continue
                splits = line.split(' ')
                sentence.append(Token(splits[0]))
                label.append(splits[-1][:-1])
        
        if len(sentence) > 0:
            yield self.text_to_instance(sentence, label)
