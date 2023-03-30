import torch
from transformers import T5ForConditionalGeneration,AutoTokenizer
import numpy as np

data = torch.LongTensor([[1, 5, 3, 23, 2], [5, 324, 12, 42, 5]])
model = T5ForConditionalGeneration.from_pretrained("t5-base")
tokenizer = AutoTokenizer.from_pretrained("t5-base")

outputs = model.generate(input_ids=data, return_dict_in_generate=True, output_scores=True)
transition_scores = model.compute_transition_scores(
    outputs.sequences, outputs.scores, normalize_logits=True
)
output_length = np.sum(transition_scores.numpy() < 0, axis=1)
length_penalty = model.generation_config.length_penalty
reconstructed_scores = transition_scores.sum(axis=1) / (output_length ** length_penalty)

# input_length is the length of the input prompt for decoder-only models, like the GPT family, and 1 for
# encoder-decoder models, like BART or T5.
input_length = 1 if model.config.is_encoder_decoder else data.shape[1]
generated_tokens = outputs.sequences[:, input_length:]
for tok, score in zip(generated_tokens[0], transition_scores[0]):
    print(f"| {tok:5d} | {tokenizer.decode(tok):8s} | {score.numpy():.3f} | {np.exp(score.numpy()):.2%}")

generate = torch.LongTensor([[0, 3, 23, 2, 3, 23, 2, 3, 23, 2, 3, 23, 2, 3,
              23, 2, 3, 23, 2, 3],
             [0, 324, 324, 324, 324, 324, 324, 324, 324, 324, 324, 324, 324, 324,
              324, 324, 324, 324, 324, 324]])
# inputs = torch.cat((data,generate),dim=1)

decoder_input_ids = model.prepare_decoder_input_ids_from_labels(generate)
outputs_forward = model(input_ids = data,decoder_input_ids=decoder_input_ids)

# outputs_forward.logits.argmax(dim=-1)
input_length = 1 if model.config.is_encoder_decoder else data.shape[1]
outputs_forward.logits = outputs_forward.logits [:,input_length:,]
transition_scores_forward = model.compute_transition_scores(
    outputs_forward.logits.argmax(dim=-1), outputs_forward.logits.split(1,dim=1), normalize_logits=True
)
output_length = np.sum(transition_scores_forward.detach().numpy() < 0, axis=1)
length_penalty = model.generation_config.length_penalty
reconstructed_scores_forward = transition_scores_forward.detach().sum(axis=1) / (output_length ** length_penalty)

generated_tokens = outputs_forward.logits.argmax(dim=-1)
print("-------------------")
for tok, score in zip(generated_tokens[0], transition_scores_forward[0]):
    print(f"| {tok:5d} | {tokenizer.decode(tok):8s} | {score.detach().numpy():.3f} | {np.exp(score.detach().numpy()):.2%}")
