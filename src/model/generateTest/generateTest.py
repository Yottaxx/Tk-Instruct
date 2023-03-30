from transformers import T5ForConditionalGeneration,AutoTokenizer
import torch

model = T5ForConditionalGeneration.from_pretrained("t5-base")
data = torch.LongTensor([[1, 5, 3, 23, 2], [5, 324, 12, 42, 5]])
model = T5ForConditionalGeneration.from_pretrained("t5-base")
tokenizer = AutoTokenizer.from_pretrained("t5-base")

outputs1 = model.generate(input_ids=data, return_dict_in_generate=True, output_scores=True,max_length=1024)
outputs2 = model.generate(input_ids=data, return_dict_in_generate=True, output_scores=True,max_length=256)

print(outputs1)