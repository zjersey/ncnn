from os import XATTR_CREATE
import transformers
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    BertModel
)
import torch
from transformers.convert_graph_to_onnx import convert
from pathlib import Path
import struct

# class ConvNet(torch.nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.conv = torch.nn.Conv2d(3, 3, (1, 1))
    
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.conv(x)
#         return x

# x = torch.rand(1, 3, 4, 4)
# net = ConvNet()
# torch.onnx.export(net, x, 'convnet.onnx')

model = BertModel.from_pretrained('bert-base-uncased').eval()

def print_hf_out():
    sentence = ["cat is a lovely girl."]
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    inputs = tokenizer(sentence, return_tensors="pt", padding=True)
    print(inputs)
    outputs = model(**inputs)

def write_bin():
    ids_load_by_1 = set([6, 8, 10, 12, 16, 18])
    with open('zzx.bin', "wb") as fw:
        for i, (k, v) in enumerate(model.named_parameters()):
            if i > 20:
                break
            lens = 0
            if i not in ids_load_by_1:
                fw.write(struct.pack('i', 0))
            for x in v.data.view(-1):
                s = struct.pack('f', x)
                fw.write(s)
                lens += 1
            print(f"{i}, {lens}")

# write_bin()
print_hf_out()