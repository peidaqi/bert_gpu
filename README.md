# BERT-GPU

A modified version of the Google BERT Tensorflow model for multi-GPU support.

You can simply use the FLAGS.use_tpu to turn GPU support on/off, either in code or command line.

For original Release Note, please refer to the [Google BERT repo]([https://github.com/google-research/bert]).

This code uses [Tensorflow distribute library](https://www.tensorflow.org/api_docs/python/tf/contrib/distribute) and thus requires [Nvidia NCCL](https://github.com/NVIDIA/nccl) to run.