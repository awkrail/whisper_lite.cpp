# Convert Hugging Face moonshine models to ggml format.
# Code is based on https://github.com/ggml-org/whisper.cpp/blob/master/models/convert-h5-to-ggml.py
#
# Usage:
#
#   cd repos
#   git clone https://huggingface.co/UsefulSensors/moonshine-tiny
#   python python/convert-h5-to-ggml.py ./repos/moonshine-tiny models

import io
import os
import sys
import struct
import json
import code
import torch
import numpy as np
from pathlib import Path

from transformers import MoonshineForConditionalGeneration

if len(sys.argv) < 3:
    print("Usage: convert-h5-to-ggml.py dir_model dir-output [use-f32]\n")
    sys.exit(1)

dir_model = Path(sys.argv[1])
dir_out = Path(sys.argv[2])

hparams = json.load((dir_model / "config.json").open("r", encoding="utf8"))
generation_hparams = json.load((dir_model / "generation_config.json").open("r", encoding="utf8"))
tokenizer = json.load((dir_model / "tokenizer.json").open("r", encoding="utf8"))

model = MoonshineForConditionalGeneration.from_pretrained(dir_model)

fname_out = dir_out / "ggml-moonshine.bin"
tokens = tokenizer['model']['vocab']

# use 16-bit or 32-bit floats
use_f16 = True
if len(sys.argv) > 3:
    use_f16 = False
    fname_out = dir_out / "ggml-model-f32.bin"

# params
fout = open(fname_out, "wb")
fout.write(struct.pack("i", 0x67676d6c)) # magic: ggml in hex
fout.write(struct.pack("i", hparams["vocab_size"]))
fout.write(struct.pack("i", hparams["hidden_size"]))
fout.write(struct.pack("i", hparams["encoder_num_attention_heads"]))
fout.write(struct.pack("i", hparams["encoder_num_hidden_layers"]))
fout.write(struct.pack("i", generation_hparams["max_length"]))
fout.write(struct.pack("i", hparams["hidden_size"]))
fout.write(struct.pack("i", hparams["decoder_num_attention_heads"]))
fout.write(struct.pack("i", hparams["decoder_num_hidden_layers"]))
fout.write(struct.pack("i", use_f16))

# tokens
fout.write(struct.pack("i", len(tokens)))
tokens = sorted(tokens.items(), key=lambda x: x[1])
for key in tokens:
    text = key[0].encode('utf-8')
    fout.write(struct.pack("i", len(text)))
    fout.write(text)

# save model params
list_vars = model.state_dict()
for name in list_vars.keys():
    data = list_vars[name].squeeze().numpy()
    data = data.astype(np.float16)

    # looks like the whisper models are in f16 by default
    # so we need to convert the small tensors to f32 until we fully support f16 in ggml
    # ftype == 0 -> float32, ftype == 1 -> float16
    n_dims = len(data.shape)
    print(name, n_dims, data.shape)

    ftype = 1
    if use_f16:
        if n_dims < 2 or \
                name == "encoder.conv1.bias"   or \
                name == "encoder.conv2.bias"   or \
                name == "encoder.positional_embedding" or \
                name == "decoder.positional_embedding":
            print("  Converting to float32")
            data = data.astype(np.float32)
            ftype = 0
    else:
        data = data.astype(np.float32)
        ftype = 0

    # header
    str_ = name.encode('utf-8')
    fout.write(struct.pack("iii", n_dims, len(str_), ftype))
    for i in range(n_dims):
        fout.write(struct.pack("i", data.shape[n_dims - 1 - i]))
    fout.write(str_)

    # data
    data.tofile(fout)

fout.close()

print("Done. Output file: " , fname_out)
print("")
