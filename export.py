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
# model = AutoModelForImageClassification.from_pretrained(
#     './beans_outputs'
# )

# for i, p in enumerate(model.parameters()):
#     print(p)
#     if i>3:
#         break

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
# model.load_state_dict(torch.load('../bert.pt'))
# sentence = ["cat is a lovely girl."]
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# inputs = tokenizer(sentence, return_tensors="pt", padding=True)
# print(inputs)
# outputs = model(**inputs) #0.1520, 0.2987
# print(outputs)
# print(outputs[0].size())
# print(model.embeddings.word_embeddings.weight.data)

# data = model.embeddings.word_embeddings.weight.data.view(-1)


with open('../zzx.bin', "wb") as fw:
    for i, (k, v) in enumerate(model.named_parameters()):
        if i > 2:
            break
        lens = 0
        fw.write(struct.pack('i', 0))
        for x in v.view(-1):
            s = struct.pack('f', i)
            fw.write(s)
            lens += 1
        print(f"{i}, {lens}")

# x = torch.LongTensor([[101, 4937, 2003, 1037, 8403, 2611, 1012,  102]])
# y = model.embeddings.word_embeddings(x)