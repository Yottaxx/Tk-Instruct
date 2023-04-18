import logging
import random
import string
from transformers.data.data_collator import *
import torch
logger = logging.getLogger(__name__)


@dataclass
class DataCollatorForRLNI:

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_source_length: Optional[int] = None
    max_target_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    add_task_name: bool = False
    add_task_definition: bool = True
    num_pos_examples: int = 0
    num_neg_examples: int = 0
    add_explanation: bool = False
    tk_instruct: bool = False
    text_only: bool=False
    

    def __call__(self, batch, return_tensors=None):

        if return_tensors is None:
            return_tensors = self.return_tensors

        sources = []

        inputs_for_instruction_generations = []
        inputs_for_answer_generation_without_instructions = []
        instructions = []
        for instance in batch:
            if self.tk_instruct:
                all_valid_encodings = [
                    # instruction only
                    {"add_task_name": False, "add_task_definition": True, "num_pos_examples": 0, "num_neg_examples": 0, "add_explanation": False}, 
                    # example only
                    {"add_task_name": False, "add_task_definition": False, "num_pos_examples": 2, "num_neg_examples": 0, "add_explanation": False}, 
                    # instruction + pos examples
                    {"add_task_name": False, "add_task_definition": True, "num_pos_examples": 2, "num_neg_examples": 0, "add_explanation": False}, 
                    # instruction + pos examples + neg examples 
                    {"add_task_name": False, "add_task_definition": True, "num_pos_examples": 2, "num_neg_examples": 2, "add_explanation": False},
                    # instruction + pos (w. explanation) 
                    {"add_task_name": False, "add_task_definition": True, "num_pos_examples": 2, "num_neg_examples": 0, "add_explanation": True}, 
                ]
                encoding_schema = random.choice(all_valid_encodings)
                add_task_name = encoding_schema["add_task_name"]
                add_task_definition = encoding_schema["add_task_definition"]
                num_pos_examples = encoding_schema["num_pos_examples"]
                num_neg_examples = encoding_schema["num_neg_examples"]
                add_explanation = encoding_schema["add_explanation"]
            else:
                add_task_name = self.add_task_name
                add_task_definition = self.add_task_definition
                num_pos_examples = self.num_pos_examples
                num_neg_examples = self.num_neg_examples
                add_explanation = self.add_explanation 

            task_input = ""
            # add the input first.
            task_input += "Now complete the following example -\n"
            task_input += f"Input: {instance['Instance']['input'].strip()}"
            if not task_input[-1] in string.punctuation:
                task_input += "."
            task_input += "\n"
            task_input += "Output: "
            task_input_for_instruction = task_input.replace("Now complete the following example -\n","")

            task_name = ""
            if add_task_name:
                task_name += instance["Task"] + ". "

            definition = ""
            if add_task_definition:
                if isinstance(instance["Definition"], list):
                    definition = "Definition: " + instance["Definition"][0].strip() # TODO: should we use <Definition>?
                else:
                    definition = "Definition: " + instance["Definition"].strip()
                if not definition[-1] in string.punctuation:
                    definition += "."
                definition += "\n\n"
            
            # try to add positive examples.
            pos_examples = []
            for idx, pos_example in enumerate(instance["Positive Examples"][:num_pos_examples]):
                pos_example_str = f" Positive Example {idx+1} -\n"
                pos_example_str += f"Input: {pos_example['input'].strip()}"
                if not pos_example_str[-1] in string.punctuation:
                    pos_example_str += "."
                pos_example_str += "\n"
                pos_example_str += f" Output: {pos_example['output'].strip()}"
                if not pos_example_str[-1] in string.punctuation:
                    pos_example_str += "."
                pos_example_str += "\n" 
                if add_explanation and "explanation" in pos_example:
                    pos_example_str += f" Explanation: {pos_example['explanation'].strip()}"
                    if not pos_example_str[-1] in string.punctuation:
                        pos_example_str += "."
                    pos_example_str += "\n"
                pos_example_str += "\n"
                if len(self.tokenizer(definition + " ".join(pos_examples) + pos_example_str + task_input)["input_ids"]) <= self.max_source_length:
                    pos_examples.append(pos_example_str)
                else:
                    break
            
            # try to add negative examples.
            neg_examples = []
            for idx, neg_example in enumerate(instance["Negative Examples"][:num_neg_examples]):
                neg_example_str = f" Negative Example {idx+1} -\n"
                neg_example_str += f"Input: {neg_example['input'].strip()}"
                if not neg_example_str[-1] in string.punctuation:
                    neg_example_str += "."
                neg_example_str += "\n"
                neg_example_str += f" Output: {neg_example['output'].strip()}"
                if not neg_example_str[-1] in string.punctuation:
                    neg_example_str += "."
                neg_example_str += "\n"
                if add_explanation and "explanation" in neg_example:
                    neg_example_str += f" Explanation: {neg_example['explanation'].strip()}"
                    if not neg_example_str[-1] in string.punctuation:
                        neg_example_str += "."
                    neg_example_str += "\n"
                neg_example_str += "\n"
                if len(self.tokenizer(definition + " ".join(pos_examples) + " ".join(neg_examples) + neg_example_str + task_input)["input_ids"]) <= self.max_source_length:
                    neg_examples.append(neg_example_str)
                else:
                    break 
            
            source = definition + task_name + "".join(pos_examples) + "".join(neg_examples) + task_input
            inputs_for_instruction_generation = "Now write a task definition for the following examples \n" + task_name + "".join(pos_examples) + "".join(neg_examples) + task_input_for_instruction
            inputs_for_answer_generation_without_instruction  = task_name + "".join(pos_examples) + "".join(neg_examples) + task_input

            tokenized_source = self.tokenizer(source)["input_ids"]
            tokenized_inputs_for_instruction_generation = self.tokenizer(inputs_for_instruction_generation)["input_ids"]
            tokenized_inputs_for_answer_generation_without_instruction = self.tokenizer(inputs_for_answer_generation_without_instruction)["input_ids"]
            tokenized_instruction = self.tokenizer(definition)["input_ids"]

            if len(tokenized_source) <= self.max_source_length:
                sources.append(source)
            else:
                sources.append(self.tokenizer.decode(tokenized_source[:self.max_source_length], skip_special_tokens=True))

            if len(tokenized_inputs_for_instruction_generation) <= self.max_source_length:
                inputs_for_instruction_generations.append(inputs_for_instruction_generation)
            else:
                inputs_for_instruction_generations.append(self.tokenizer.decode(tokenized_source[:self.max_source_length], skip_special_tokens=True))

            if len(tokenized_inputs_for_answer_generation_without_instruction) <= self.max_source_length:
                inputs_for_answer_generation_without_instructions.append(inputs_for_answer_generation_without_instruction)
            else:
                inputs_for_answer_generation_without_instructions.append(self.tokenizer.decode(tokenized_source[:self.max_source_length], skip_special_tokens=True))

            if len(tokenized_instruction) <= self.max_source_length:
                instructions.append(definition)
            else:
                instructions.append(self.tokenizer.decode(tokenized_instruction[:self.max_source_length], skip_special_tokens=True))

        if self.text_only:
            model_inputs = {"inputs": sources}
        else:
            model_inputs = self.tokenizer(
                sources, 
                max_length=self.max_source_length, 
                padding=self.padding,
                return_tensors=self.return_tensors, 
                truncation=True,
                pad_to_multiple_of=self.pad_to_multiple_of)

            inputs_for_instruction_generations = self.tokenizer(
                inputs_for_instruction_generations,
                max_length=self.max_source_length,
                padding=self.padding,
                return_tensors=self.return_tensors,
                truncation=True,
                pad_to_multiple_of=self.pad_to_multiple_of)

            inputs_for_answer_generation_without_instructions = self.tokenizer(
                inputs_for_answer_generation_without_instructions,
                max_length=self.max_source_length,
                padding=self.padding,
                return_tensors=self.return_tensors,
                truncation=True,
                pad_to_multiple_of=self.pad_to_multiple_of)

            with self.tokenizer.as_target_tokenizer():
                instruction_label = self.tokenizer(
                    instructions,
                    max_length=self.max_target_length,
                    padding=self.padding,
                    return_tensors=self.return_tensors,
                    truncation=True,
                    pad_to_multiple_of=self.pad_to_multiple_of
                )

                if len(instruction_label["input_ids"])<self.max_target_length:
                    instruction_label = self.tokenizer.pad(
                        instruction_label,
                        padding="max_length",
                        max_length=self.max_target_length,
                        pad_to_multiple_of=self.pad_to_multiple_of,
                        return_tensors=return_tensors,
                    )

            model_inputs["instructed_input_ids"] = model_inputs["input_ids"]
            model_inputs["instructed_attention_mask"] = model_inputs["attention_mask"]

            model_inputs["input_ids"] = inputs_for_instruction_generations["input_ids"]
            model_inputs["attention_mask"] = inputs_for_instruction_generations["attention_mask"]

            model_inputs["woinstruction_input_ids"] = inputs_for_answer_generation_without_instructions["input_ids"]
            model_inputs["woinstruction_mask"] = inputs_for_answer_generation_without_instructions["attention_mask"]


            label_mask = instruction_label["attention_mask"].bool()
            model_inputs["instruction_labels"] = instruction_label["input_ids"].masked_fill(~label_mask, self.label_pad_token_id)

        if "output" in batch[0]["Instance"] and batch[0]["Instance"]["output"]:
            # Randomly select one reference if multiple are provided.
            labels = [random.choice(ex["Instance"]["output"]) for ex in batch]
            if self.text_only:
                model_inputs["labels"] = labels
            else:
                with self.tokenizer.as_target_tokenizer():
                    labels = self.tokenizer(
                        labels,
                        max_length=self.max_target_length,
                        padding=self.padding,
                        return_tensors=self.return_tensors,
                        truncation=True,
                        pad_to_multiple_of=self.pad_to_multiple_of
                    )

                if len(instruction_label["input_ids"])<self.max_target_length:
                    labels = self.tokenizer.pad(
                        labels,
                        padding="max_length",
                        max_length=self.max_target_length,
                        pad_to_multiple_of=self.pad_to_multiple_of,
                        return_tensors=return_tensors,
                    )

                label_mask = labels["attention_mask"].bool()
                model_inputs["labels"] = labels["input_ids"].masked_fill(~label_mask, self.label_pad_token_id)
        else:
            model_inputs["labels"] = None

        # prepare decoder_input_ids
        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels") and not self.text_only:
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=model_inputs["labels"])
            instruction_decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=model_inputs["instruction_labels"])

            model_inputs["decoder_input_ids"] = decoder_input_ids
            model_inputs["instruction_decoder_input_ids"] = instruction_decoder_input_ids

        return model_inputs


@dataclass
class DataCollatorForRLExNI:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_source_length: Optional[int] = None
    max_target_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, batch, return_tensors=None):

        instances =  [batch[i]["instance"][0] for i in range(len(batch))]
        items =  [batch[i]["item"][0] for i in range(len(batch))]
        instructions =  torch.LongTensor([batch[i]["instruction"] for i in range(len(batch))]).squeeze(dim=1)
        labels =  torch.LongTensor([batch[i]["labels"] for i in range(len(batch))])

        reward =  torch.Tensor([batch[i]["reward"] for i in range(len(batch))])
        logits =  torch.Tensor([batch[i]["logits"] for i in range(len(batch))])
        value = torch.Tensor([batch[i]["value"] for i in range(len(batch))])
        ref_logits = torch.Tensor([batch[i]["ref_logits"] for i in range(len(batch))])
        logits_mask =  torch.Tensor([batch[i]["logits_mask"] for i in range(len(batch))])

        model_inputs = self.tokenizer(
            instances,
            max_length=self.max_source_length,
            padding=self.padding,
            return_tensors=self.return_tensors,
            truncation=True,
            pad_to_multiple_of=self.pad_to_multiple_of)

        instructed_input = self.tokenizer(
            items,
            max_length=self.max_source_length,
            padding=self.padding,
            return_tensors=self.return_tensors,
            truncation=True,
            pad_to_multiple_of=self.pad_to_multiple_of)

        model_inputs["instructed_input_ids"] = instructed_input["input_ids"]
        model_inputs["instructed_attention_mask"] = instructed_input["attention_mask"]

        model_inputs["instruction_labels"] = instructions
        model_inputs["labels"] = labels

        # prepare decoder_input_ids
        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=model_inputs["labels"])
            instruction_decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=model_inputs["instruction_labels"])

            model_inputs["decoder_input_ids"] = decoder_input_ids
            model_inputs["instruction_decoder_input_ids"] = instruction_decoder_input_ids


        model_inputs["old_actions_log_probs"] = logits
        model_inputs["old_ref_log_probs"] = ref_logits
        model_inputs["old_rewards"] =  reward
        model_inputs["old_values"] =  value
        model_inputs["old_actions_log_probs_mask"] =  logits_mask
        return model_inputs