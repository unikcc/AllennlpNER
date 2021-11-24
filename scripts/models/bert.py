from typing import Any, Dict, Optional, Union

import numpy
from overrides import overrides
import torch
from torch import nn
import torch.nn.functional as F

from allennlp.common.checks import check_dimensions_match, ConfigurationError
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.modules import FeedForward, Maxout, Seq2SeqEncoder, TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy

from scripts.training.NERMetrics import NERMetrics

@Model.register("bert")
class LSTMClassifier(Model):

    def __init__(
        self,
        vocab: Vocabulary,
        embedder: Optional[Dict[str, Any]],
        embedding_dropout: float,
        pre_encode_feedforward: FeedForward,
        encoder: Seq2SeqEncoder,
        integrator: Seq2SeqEncoder,
        integrator_dropout: float,
        projection_layer: FeedForward,
        initializer: InitializerApplicator = InitializerApplicator(),
        **kwargs,
    ) -> None:
        super().__init__(vocab, **kwargs)

        
        self._embedding_dropout = nn.Dropout(embedding_dropout)
        self._num_classes = self.vocab.get_vocab_size("ner_tags")
        # print(self.vocab.get_index_to_token_vocabulary('tokens'))

        self._pre_encode_feedforward = pre_encode_feedforward
        self._encoder = encoder
        self._integrator = integrator
        self._integrator_dropout = nn.Dropout(integrator_dropout)

       
        self._combined_integrator_output_dim = self._integrator.get_output_dim()

        self.projection_layer = projection_layer
        output_dim = self.projection_layer.get_output_dim()

        self._self_attentive_pooling_projection = nn.Linear(self._combined_integrator_output_dim, 1)
        self.output_layer = nn.Linear(output_dim, self._num_classes) 

        index2token = vocab.get_index_to_token_vocabulary('ner_tags')
        self._metric = NERMetrics(index2token)

        check_dimensions_match(
            embedder['output_dim'],
            self._pre_encode_feedforward.get_input_dim(),
            "text field embedder output dim",
            "Pre-encoder feedforward input dim",
        )

        check_dimensions_match(
            self._pre_encode_feedforward.get_output_dim(),
            self._encoder.get_input_dim(),
            "Pre-encoder feedforward output dim",
            "Encoder input dim",
        )
        check_dimensions_match(
            self._encoder.get_output_dim() * 3,
            self._integrator.get_input_dim(),
            "Encoder output dim * 3",
            "Integrator input dim",
        )
        check_dimensions_match(
            self._integrator.get_output_dim(),
            self.projection_layer.get_input_dim(),
            "Integrator output dim * 4",
            "Output layer input dim",
        )

        self.loss = torch.nn.CrossEntropyLoss()
        initializer(self)

    @overrides
    def forward(
        self,  # type: ignore
        tokens: TextFieldTensors,
        label: torch.LongTensor = None,
    ) -> Dict[str, torch.Tensor]:
        # x = self.training
        # print(x)

        text_mask = util.get_text_field_mask(tokens)
        # print(text_mask)
        if tokens:
            embedded_text = self._text_field_embedder(tokens)

        dropped_embedded_text = self._embedding_dropout(embedded_text)
        pre_encoded_text = self._pre_encode_feedforward(dropped_embedded_text)
        encoded_tokens = self._encoder(pre_encoded_text, text_mask)

        # Compute biattention. This is a special case since the inputs are the same.
        attention_logits = encoded_tokens.bmm(encoded_tokens.permute(0, 2, 1).contiguous())
        attention_weights = util.masked_softmax(attention_logits, text_mask)
        encoded_text = util.weighted_sum(encoded_tokens, attention_weights)

        # Build the input to the integrator
        integrator_input = torch.cat(
            [encoded_tokens, encoded_tokens - encoded_text, encoded_tokens * encoded_text], 2
        )
        integrated_encodings = self._integrator(integrator_input, text_mask)

        pooled_representations_dropped = self._integrator_dropout(integrated_encodings)

        projection_output = self.projection_layer(pooled_representations_dropped)

        logits = self.output_layer(projection_output)

        class_probabilities = F.softmax(logits, dim=-1)

        output_dict = {"logits": logits, "class_probabilities": class_probabilities}
        if label is not None:
            active_loss = text_mask.view(-1) == 1
            active_logits = logits.view(-1, self._num_classes)[active_loss]
            active_labels = label.view(-1)[active_loss]
            loss = self.loss(active_logits, active_labels)

            self._metric(logits.argmax(-1), label, text_mask)

            output_dict["loss"] = loss

        return output_dict

    @overrides
    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the class probabilities, converts indices to string labels, and
        adds a `"label"` key to the dictionary with the result.
        """
        predictions = output_dict["class_probabilities"].cpu().data.numpy()
        argmax_indices = numpy.argmax(predictions, axis=-1)
        labels = [self.vocab.get_token_from_index(x, namespace="labels") for x in argmax_indices]
        output_dict["label"] = labels
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            metric_name: score for metric_name, score in self._metric.get_metric().items()
        }
