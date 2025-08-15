import torch
from transformers import LlamaConfig

from QEfficient.transformers.cache_utils import QEffDynamicCache
from QEfficient.transformers.models.llama.modeling_llama import QEffLlamaForCausalLM


def test_prefill_queries_shape():
    config = LlamaConfig(
        vocab_size=100,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
    )
    model = QEffLlamaForCausalLM(config)
    input_ids = torch.randint(0, config.vocab_size, (1, 4))
    position_ids = torch.arange(0, input_ids.shape[1]).unsqueeze(0)
    outputs = model(
        input_ids=input_ids,
        position_ids=position_ids,
        past_key_values=QEffDynamicCache(),
        use_cache=True,
        return_dict=True,
    )
    prefill = getattr(outputs, "prefill_queries", None)
    assert prefill is not None
    assert prefill.shape == (
        config.num_hidden_layers,
        config.num_attention_heads,
        config.hidden_size // config.num_attention_heads,
    )

