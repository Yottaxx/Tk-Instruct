from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained(
    "t5-base",
)

peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
)

p1 = get_peft_model(model, peft_config)
p2 = get_peft_model(model, peft_config)
print("yes")
print(p2.base_model.model.encoder.block[0].layer[0].SelfAttention.q.lora_A.weight.data_ptr())
print(p1.base_model.model.encoder.block[0].layer[0].SelfAttention.q.lora_A.weight.data_ptr())