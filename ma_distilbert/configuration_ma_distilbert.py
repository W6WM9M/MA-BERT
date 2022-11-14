# coding=utf-8
# Copyright 2019-present, the HuggingFace Inc. team, The Google AI Language Team and Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

### Note that MA-DistilBERT is simply DistilBERT with just several lines of modifications.
### We are simply reusing DistilBERT's source code for our purpose. 
""" MA-DistilBERT model configuration"""

from collections import OrderedDict
from typing import Mapping

from ....configuration_utils import PretrainedConfig
from ....onnx import OnnxConfig
from ....utils import logging


logger = logging.get_logger(__name__)

DISTILBERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "distilbert-base-uncased": "https://huggingface.co/distilbert-base-uncased/resolve/main/config.json",
    "distilbert-base-uncased-distilled-squad": "https://huggingface.co/distilbert-base-uncased-distilled-squad/resolve/main/config.json",
    "distilbert-base-cased": "https://huggingface.co/distilbert-base-cased/resolve/main/config.json",
    "distilbert-base-cased-distilled-squad": "https://huggingface.co/distilbert-base-cased-distilled-squad/resolve/main/config.json",
    "distilbert-base-german-cased": "https://huggingface.co/distilbert-base-german-cased/resolve/main/config.json",
    "distilbert-base-multilingual-cased": "https://huggingface.co/distilbert-base-multilingual-cased/resolve/main/config.json",
    "distilbert-base-uncased-finetuned-sst-2-english": "https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english/resolve/main/config.json",
}

### Class name modified to prevent conflict with original DistilBERT 
class MA_DistilBertConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`DistilBertModel`] or a [`TFDistilBertModel`]. It
    is used to instantiate a DistilBERT model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the DistilBERT
    [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the DistilBERT model. Defines the number of different tokens that can be represented by
            the `inputs_ids` passed when calling [`DistilBertModel`] or [`TFDistilBertModel`].
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        sinusoidal_pos_embds (`boolean`, *optional*, defaults to `False`):
            Whether to use sinusoidal positional embeddings.
        n_layers (`int`, *optional*, defaults to 6):
            Number of hidden layers in the Transformer encoder.
        n_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        dim (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        hidden_dim (`int`, *optional*, defaults to 3072):
            The size of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        activation (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        qa_dropout (`float`, *optional*, defaults to 0.1):
            The dropout probabilities used in the question answering model [`DistilBertForQuestionAnswering`].
        seq_classif_dropout (`float`, *optional*, defaults to 0.2):
            The dropout probabilities used in the sequence classification and the multiple choice model
            [`DistilBertForSequenceClassification`].

    Examples:

    ```python
    >>> from transformers import DistilBertModel, DistilBertConfig

    >>> # Initializing a DistilBERT configuration
    >>> configuration = DistilBertConfig()

    >>> # Initializing a model from the configuration
    >>> model = DistilBertModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "distilbert"
    attribute_map = {
        "hidden_size": "dim",
        "num_attention_heads": "n_heads",
        "num_hidden_layers": "n_layers",
    }

    def __init__(
        self,
        vocab_size=30522,
        max_position_embeddings=512,
        sinusoidal_pos_embds=False,
        n_layers=6,
        n_heads=12,
        dim=768,
        hidden_dim=4 * 768,
        dropout=0.1,
        attention_dropout=0.1,
        cls_activation="gelu",
        # Relu Modification
        ffn_activation="relu",
        initializer_range=0.02,
        qa_dropout=0.1,
        seq_classif_dropout=0.2,
        pad_token_id=0,
        # PowerNorm
        norm_type="Power",
        warmup_updates=997120, ### Expected total gradient updates * accumulation step 
        accumulation_step=8,### Adapt PowerNorm source code for gradient accumulation
        layer_norm_eps=1e-12,
        # Softmax Approximation
        use_softmax_approx=True,
        share_softmax_nn=False,
        softmax_input_size = 128,
        softmax_hidden_layer_size = 128,
        # Knowledge Transfer
        output_hidden_states = True, 
        output_attentions = True,
        verbose = False,
        **kwargs
    ):
        super().__init__(**kwargs, pad_token_id=pad_token_id)

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.sinusoidal_pos_embds = sinusoidal_pos_embds
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.cls_activation = cls_activation

        ### For replacing GELU with ReLU
        self.ffn_activation = ffn_activation

        self.initializer_range = initializer_range
        self.qa_dropout = qa_dropout
        self.seq_classif_dropout = seq_classif_dropout

        ### For replacing LayerNorm with PowerNorm
        self.norm_type = norm_type
        self.warmup_updates = warmup_updates
        self.accumulation_step = accumulation_step
        self.layer_norm_eps = layer_norm_eps

        ### For replacing Softmax with 2-layer neural network approximation
        self.use_softmax_approx = use_softmax_approx
        self.share_softmax_nn = share_softmax_nn
        self.softmax_input_size = softmax_input_size
        self.softmax_hidden_layer_size = softmax_hidden_layer_size
        

        ### For Knowledge Transfer
        self.output_hidden_states = output_hidden_states
        self.output_attentions = output_attentions
        
        if verbose:
            print(f"Modifications Applied:")
            print(f"Softmax Approximation: {self.use_softmax_approx}, Share Softmax: {self.share_softmax_nn}, Input Size: {self.softmax_input_size}, Hidden Size: {self.softmax_hidden_layer_size}")
            print(f"Normalization: {self.norm_type}, Warm-up Iterations: {self.warmup_updates}, Accumulation Steps: {self.accumulation_step}")
            print(f"Encoder Activation Function: {self.ffn_activation}")


class DistilBertOnnxConfig(OnnxConfig):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        if self.task == "multiple-choice":
            dynamic_axis = {0: "batch", 1: "choice", 2: "sequence"}
        else:
            dynamic_axis = {0: "batch", 1: "sequence"}
        return OrderedDict(
            [
                ("input_ids", dynamic_axis),
                ("attention_mask", dynamic_axis),
            ]
        )
