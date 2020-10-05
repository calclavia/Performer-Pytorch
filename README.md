# Performer-Pytorch
[WIP]

Pytorch implementation of Performer self attention module from the paper ["Rethinking Attention with Performers"](https://arxiv.org/abs/2009.14794).

This implementation is based on available code in https://github.com/google-research/google-research/tree/master/performer/fast_self_attention

## Test
Run tests to compare against Google's implementation in JAX.
```
python test.py
```
Requires JAX to be installed.