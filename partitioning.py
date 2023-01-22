import re

from flax.core.frozen_dict import freeze
from flax.traverse_util import flatten_dict, unflatten_dict
from jax.experimental import PartitionSpec as P

# utils adapted from https://github.com/google-research/google-research/blob/master/flax_models/t5x/partitions.py
# Sentinels
_unmatched = object()

# For specifying empty leaf dict `{}`
empty_dict = object()


def _match(qs, ks):
    """Return True if regexes in qs match any window of strings in tuple ks."""
    # compile regexes and force complete match
    qts = tuple(map(lambda x: re.compile(x + "$"), qs))
    for i in range(len(ks) - len(qs) + 1):
        matches = [x.match(y) for x, y in zip(qts, ks[i:])]
        if matches and all(matches):
            return True
    return False


def _replacement_rules(rules):
    def replace(key, val):
        for rule, replacement in rules:
            if _match(rule, key):
                return replacement
        return val

    return replace


def __get_partition_rules():
    return [
        # embeddings
        (("embed_positions", "embedding"), P("mp", None)),
        (("embed_tokens", "embedding"), P("mp", None)),
        (("rel_bias", "embedding"), P(None, "mp")),
        # attention
        (("(q_proj|k_proj|v_proj)", "kernel"), P(None, "mp")),
        (("out_proj", "kernel"), P("mp", None)),
        # FFN
        (("Dense_0", "kernel"), P(None, "mp")),
        (("GLU.*", "Dense_1", "kernel"), P(None, "mp")),
        (("GLU.*", "Dense_2", "kernel"), P("mp", None)),
        (("FFN.*", "Dense_1", "kernel"), P("mp", None)),
        # layer norms
        (("(bias|scale)",), None),
        (("lm_head", "kernel"), P(None, "mp")),
        # head scale and tau
        (("(head_scale|tau)",), None),
    ]

def _get_partition_rules():
    return [
        # embeddings
        (("params", "Conv_0", "kernel"), P(None, None)),
        (("params", "Conv_0", "bias"), P(None,)),

        (("params", "CrossEmbedLayer_0", "Conv_.*", "kernel"), P(None, None, None, "mp")),
        (("params", "CrossEmbedLayer_0", "Conv_.*", "bias"), P("dp", )),
        
        (("params", "LearnedSinusoidalPosEmb_0", "pos_emb"), P("mp",)),
        
        (("params", "Dense_.*", "kernel"), P(None, "mp")), # time stuff
        (("params", "Dense_.*", "bias"), P("dp", )),
        
        (("params", "TextConditioning_.*", "Dense_0", "kernel"), P(None, "mp")), # text_tokens
        (("params", "TextConditioning_.*", "null_text_embed"), P(None, "mp")),
        (("params", "TextConditioning_.*", "Dense_1", "kernel"), P("mp", None)), # text_hidden
        (("params", "TextConditioning_.*", "Dense_2", "kernel"), P(None, "mp")), # text_hiddens 2
        (("params", "TextConditioning_.*", "Dense_.*", "bias"), P("dp", )),
        (("params", "TextConditioning_.*", "LayerNorm_0", "bias|scale"), P("dp", )),
        (("params", "TextConditioning_.*", "LayerNorm_1", "bias|scale"), P("dp", )),
        (("params", "TextConditioning_.*", "null_text_hidden"), P(None, "mp")),

        
        (("params", "Downsample_0", "Conv_0", "kernel"), P(None, None, "mp", "dp")),
        (("params", "Downsample_0", "Conv_0", "bias"), P("dp",)),
        
        (('params', 'Attention_.*', 'Dense_.*', 'kernel'), P(None, "mp")),
        (('params', 'Attention_.*', 'LayerNorm_.*', 'g'), P("dp")),
        (('params', 'Attention_.*', 'null_kv'), P("dp", "mp")),
        
        (('params', 'TransformerBlock_.*', 'Attention_.*', 'Dense_.*', 'kernel'), P(None, "mp")),
        (('params', 'TransformerBlock_.*', 'Attention_.*', 'LayerNorm_.*', 'g'), P("dp")),
        (('params', 'TransformerBlock_.*','Attention_.*', 'null_kv'), P("dp", "mp")),
        
        
        (("params", "(UpsampleCombiner_.*|ResnetBlock_.*)", "Block_.*",  "Conv_0", "kernel"), P(None, None, "mp", "dp")),
        (("params", "(UpsampleCombiner_.*|ResnetBlock_.*)", "Block_.*",  "Conv_0", "bias"), P("dp",)),
        (("params", "(UpsampleCombiner_.*|ResnetBlock_.*)", "Block_.*",  "GroupNorm_0", "bias|scale"), P("dp", )),
        
        (("params", "ResnetBlock_.*", "Conv_0", "kernel"), P(None, None, "mp", "dp")),
        (("params", "ResnetBlock_.*", "Conv_0", "bias"), P("dp",)),
        (("params", "ResnetBlock_.*", "Dense_0", "kernel"), P("mp", None)),
        (("params", "ResnetBlock_.*", "Dense_0", "bias"), P("dp", )),
        (("params", "ResnetBlock_.*", "CrossAttention_.*", "Dense_.*","kernel"), P(None, "mp")),
        (("params", "ResnetBlock_.*", "CrossAttention_.*", "null_kv"), P("dp", "mp")),
        (("params", "ResnetBlock_.*", "CrossAttention_.*", "LayerNorm_.*","bias|scale"), P("dp", )),
                        
        (("params", "Upsample_0", "Conv_0", "kernel"), P(None, None, "mp", "dp")),
        (("params", "Upsample_0", "Conv_0", "bias"), P("dp", )),
    ]


def set_partitions(in_dict):
    #TODO: Use scan
    rules = _get_partition_rules()
    replace = _replacement_rules(rules)
    initd = {k: _unmatched for k in flatten_dict(in_dict)}
    result = {k: replace(k, v) for k, v in initd.items()}
    for k, v in result.items():
        if v == _unmatched:
            print(f"Unmatched -> {k}")
    l = list(result.keys())
    assert _unmatched not in result.values(), "Incomplete partition spec." 
    return freeze(unflatten_dict(result))
