import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

# load a T5-small model
model = T5ForConditionalGeneration.from_pretrained('t5-base').to(device)
tokenizer = T5Tokenizer.from_pretrained('t5-base', model_max_length=512)
model.eval()

# define some source text and tokenize it
source_text = "This is a source sentence."
source_ids = tokenizer(source_text, return_tensors="pt").input_ids.to(device)

# generate the output using beam search
gen_outputs = model.generate(
    inputs=source_ids,
    num_beams=3,
    min_length=0,
    max_length=512,
    length_penalty=0,
    output_scores=True,
    output_hidden_states = True,
    return_dict_in_generate=True,
)

# compute the scores using compute_transition_scores()
scores = model.compute_transition_scores(
    sequences=gen_outputs.sequences,
    scores=gen_outputs.scores,
    beam_indices=gen_outputs.beam_indices,
    # (normalize_logits=True) for greedy
)

# compute the loss for the generated sequence
outputs = model(
    input_ids=source_ids,
    attention_mask=torch.ones_like(source_ids),
    labels=gen_outputs.sequences[:, 1:],  # skip BOS token
    return_dict=True
)
loss = outputs.loss.item()

# compare the scores given by generate() with the loss given by forward()
print('scores (generate):', gen_outputs.sequences_scores.item())
print('scores (compute_transition_scores):', scores.sum().item())
print('loss * seq_len:', loss * (gen_outputs.sequences.shape[-1] - 1))  # correct length
print('loss * seq_len:', loss * ((gen_outputs.sequences[:,1:]!=-100).sum(dim=-1)))  # correct length

print('loss:', loss)