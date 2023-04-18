from dataclasses import dataclass
from typing import Tuple, Optional

import torch
from peft import get_peft_model, TaskType, LoraConfig
from torch.nn import CrossEntropyLoss, KLDivLoss
from transformers import AutoModelForSeq2SeqLM
from transformers.utils import ModelOutput
import torch.nn.functional as F


@dataclass
class RLSeq2SeqLMOutput(ModelOutput):
    """
    Base class for sequence-to-sequence language models outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
        decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
        encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    prompt_logits: Optional[torch.FloatTensor] = None
    value_logits: Optional[torch.FloatTensor] = None
    policy_loss: Optional[torch.FloatTensor] = None
    kl_loss: Optional[torch.FloatTensor] = None
    critic_loss: Optional[torch.FloatTensor] = None


class ActorCritic(torch.nn.Module):
    """Actor Critic class stores both the actor and the critic models
    and it generates values and action for given sequences during the training
    of the actor.

    Attributes:
        actor (ActorModel): Actor model
        critic (CriticModel): Critic model
        debug (bool): enable prints for Debugging

    Methods:
        forward: given a sequence returns action logits and values (used
            to evaluate the actor during training)
        generate: given a sequence returns action, action logits, values
            sequences and sequences masks (used to generate new sequences
            during acting phase)
    """

    def __init__(
            self, model_args, config, peft_config
    ) -> None:
        super().__init__()


        actor = AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
        )
        actor_ema = AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
        )
        self.actor = get_peft_model(actor, peft_config)
        self.actor_ema = get_peft_model(actor_ema, peft_config)

        critic = AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
        )
        self.critic = get_peft_model(critic, peft_config)

        self.ppo_beam = model_args.ppo_beam
        self.config = config

        self.ref_model = AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
        )

        self.reward_model = AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
        )
        # self.ref_model = get_peft_model(ref_model, peft_config)
        # self.reward_model = get_peft_model(reward_model, peft_config)
        for param in self.actor_ema.parameters():
            param.requires_grad = False

        for param in self.reward_model.parameters():
            param.requires_grad = False

        for param in self.ref_model.parameters():
            param.requires_grad = False

        self.kl_ctl = 0.02
        self.clip_reward_value = 5
        self.cliprange = 0.2
        self.cliprange_value = 0.2
        self.gamma = 1.0
        self.lam = 0.95
        self.PAD_ID = 0

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self.critic._shift_right(labels)

    def compute_rewards(self, log_probs, ref_log_probs, reward_score,
                        action_mask):

        kl_divergence_estimate = -self.kl_ctl * (log_probs - ref_log_probs)
        rewards = kl_divergence_estimate
        start = 0
        ends = start + action_mask[:, start:].sum(1)
        reward_clip = torch.clamp(reward_score, -self.clip_reward_value,
                                  self.clip_reward_value)
        batch_size = log_probs.shape[0]
        for j in range(batch_size):
            rewards[j, start:ends[j]][-1] += reward_clip[j]

        return rewards, kl_divergence_estimate.mean()

    def actor_loss_fn(self, logprobs, old_logprobs, advantages, mask):
        ## policy gradient loss
        log_ratio = (logprobs - old_logprobs) * mask
        ratio = torch.exp(log_ratio)
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - self.cliprange,
                                             1.0 + self.cliprange)
        pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) / mask.sum()
        return pg_loss

    def critic_loss_fn(self, values, old_values, returns, mask):
        ## value loss
        values_clipped = torch.clamp(
            values,
            old_values - self.cliprange_value,
            old_values + self.cliprange_value,
        )
        vf_loss1 = (values - returns) ** 2
        vf_loss2 = (values_clipped - returns) ** 2
        vf_loss = 0.5 * torch.sum(
            torch.max(vf_loss1, vf_loss2) * mask) / mask.sum()
        return vf_loss

    def get_advantages_and_returns(self, values, rewards, start):
        # Adopted from https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py#L134
        lastgaelam = 0
        advantages_reversed = []
        length = rewards.size()[-1]
        for t in reversed(range(start, length)):
            nextvalues = values[:, t + 1] if t < length - 1 else 0.0
            delta = rewards[:, t] + self.gamma * nextvalues - values[:, t]
            lastgaelam = delta + self.gamma * self.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values[:, start:]
        return advantages.detach(), returns

    def gather_log_probs(self, logits, labels):
        log_probs = F.log_softmax(logits, dim=-1)
        log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
        return log_probs_labels.squeeze(-1)

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            instructed_input_ids: Optional[torch.LongTensor] = None,
            instructed_attention_mask: Optional[torch.LongTensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.BoolTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            decoder_head_mask: Optional[torch.FloatTensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            old_actions_log_probs: Optional[torch.Tensor] = None,
            old_ref_log_probs: Optional[torch.Tensor] = None,
            old_rewards: Optional[torch.Tensor] = None,
            old_values: Optional[torch.Tensor] = None,
            old_actions_log_probs_mask: Optional[torch.Tensor] = None,
            instruction_labels: Optional[torch.Tensor] = None,
            instruction_decoder_input_ids: Optional[torch.Tensor] = None,
            woinstruction_input_ids: Optional[torch.Tensor] = None,
            woinstruction_mask: Optional[torch.Tensor] = None,
    ) -> RLSeq2SeqLMOutput:
        """Given the whole sequences, use the actor forward to get the logits
            for each token in the sequence and the critic forward to get the
            values for each generation step.

        Args:
            sequences (torch.Tensor): Sequences composed of [states, actions]
            sequence_mask (torch.Tensor): Mask for the sequences
            action_length (int): Length of the actions in the sequences

        Returns:
            action_logits (torch.Tensor): Logits for the actions in the
                sequences
            values (torch.Tensor): Values for the actions in the sequences
        """
        # use a single forward on the whole sequence
        # to get pi(y | x) and ignore predicted output
        output_actor = self.actor(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=instruction_decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=instruction_labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        output_critic = self.critic(
            input_ids=instructed_input_ids,
            attention_mask=instructed_attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            # labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict, )

        if not self.training:
            return RLSeq2SeqLMOutput(
                loss=torch.Tensor([0.0]).to(input_ids.device()),
                prompt_logits=output_actor.logits,
                value_logits=output_critic.logits
            )

        loss_fct = CrossEntropyLoss(ignore_index=-100, reduction="none")
        action_mask = torch.bitwise_and(instruction_labels != -100, old_actions_log_probs_mask.bool())
        with torch.no_grad():
            old_rewards, kl_divergence_estimate = self.compute_rewards(old_actions_log_probs, old_ref_log_probs,
                                                                       old_rewards, action_mask)
            advantages, returns = self.get_advantages_and_returns(
                old_values, old_rewards, 0)
        actor_prob = output_actor.logits
        actor_log_prob = self.gather_log_probs(actor_prob[:, :-1, :], instruction_decoder_input_ids[:, 1:])
        actor_loss = self.actor_loss_fn(actor_log_prob, old_actions_log_probs, advantages, action_mask)

        value = -loss_fct(output_critic.logits.view(-1, output_critic.logits.size(-1)), labels.view(-1))[:, :-1]
        critic_loss = self.critic_loss_fn(value, old_values, returns, action_mask)

        return RLSeq2SeqLMOutput(
            loss=actor_loss + critic_loss,
            prompt_logits=output_actor.logits,
            value_logits=value,
            policy_loss=actor_loss,
            kl_loss=kl_divergence_estimate,
            critic_loss=critic_loss
        )

    def eval_state(self):
        self.actor.eval()
        self.critic.eval()
        self.reward_model.eval()
        self.ref_model.eval()

    def train_state(self):
        self.actor.train()
        self.critic.train()

    @torch.no_grad()
    def generate_ppo(
            self,
            input_ids: torch.Tensor = None,
            attention_mask: torch.Tensor = None,
            instructed_input_ids=None,
            instructed_attention_mask=None,
            decoder_input_ids=None,
            labels=None,
            woinstruction_input_ids=None,
            woinstruction_mask=None,
            instruction_labels=None,
            instruction_decoder_input_ids=None,
            **kwargs,
    ):

        kwargs["input_ids"] = input_ids
        kwargs["attention_mask"] = attention_mask
        kwargs["output_scores"] = True
        kwargs["return_dict_in_generate"] = True
        kwargs["temperature"] = 0.9
        kwargs["max_length"] = self.logits_shape - 1
        # kwargs["num_beams"] = self.ppo_beam
        # kwargs["num_return_sequences"] = self.ppo_beam
        # kwargs["do_sample"] = True

        # for peft
        output_actor = self.actor.generate(
            **kwargs
        )

        return output_actor.sequences

    @torch.no_grad()
    def get_actor_log_prob(self, input_ids=None,
                           attention_mask=None,
                           decoder_input_ids=None,
                           labels=None
                           ):

        if decoder_input_ids is None:
            decoder_input_ids = self.actor.prepare_decoder_input_ids_from_labels(labels)

        output_actor = self.actor(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels
        )
        return self.gather_log_probs(output_actor.logits[:, :-1, :], decoder_input_ids[:, 1:])

    @torch.no_grad()
    def get_ref_log_prob(self, input_ids=None,
                           attention_mask=None,
                           decoder_input_ids=None,
                           labels=None
                           ):

        if decoder_input_ids is None:
            decoder_input_ids = self.ref_model.prepare_decoder_input_ids_from_labels(labels)

        output_actor = self.ref_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels
        )
        return self.gather_log_probs(output_actor.logits[:, :-1, :], decoder_input_ids[:, 1:])

    @torch.no_grad()
    def get_reward(self, input_ids=None,
                           attention_mask=None,
                           decoder_input_ids = None,
                           labels = None,
                            return_value_only=False
                           ):

        if decoder_input_ids is None:
            decoder_input_ids = self.reward_model.prepare_decoder_input_ids_from_labels(labels)

        output_critic = self.reward_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
        )
        loss_fct = CrossEntropyLoss(ignore_index=-100, reduction="none")
        values = loss_fct(output_critic.logits.view(-1, output_critic.logits.size(-1)), labels.view(-1))

        if return_value_only:
            return values
        else:
            # [0 0 0 0 prompt, answer, 0 0 0 0 ] for step 3, we have padding at the beginning
            # [prompt, answer, 0, 0, 0, 0] this is normal
            bs = values.size(0)
            seq_len = input_ids.shape[1]
            chosen_end_scores = [
            ]  # we use this name for consistency with the original forwad function
            for i in range(bs):
                input_id = input_ids[i]
                value = values[i]

                c_inds = (input_id == self.PAD_ID).nonzero()
                # here we only use the answer part of the sequence so we do not need to care about the padding at the beginning
                c_ind = c_inds[0].item() if len(
                    c_inds) > 0 else seq_len
                chosen_end_scores.append(value[c_ind - 1])
            return {
                "values": values,
                "chosen_end_scores": torch.stack(chosen_end_scores),
            }

    @torch.no_grad()
    def get_critic(self, input_ids=None,
                            attention_mask=None,
                            decoder_input_ids=None,
                            labels=None,
                            return_value_only=False
                            ):

        if decoder_input_ids is None:
            decoder_input_ids = self.critic.prepare_decoder_input_ids_from_labels(labels)

        output_critic = self.critic(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
        )
        loss_fct = CrossEntropyLoss(ignore_index=-100, reduction="none")
        values = loss_fct(output_critic.logits.view(-1, output_critic.logits.size(-1)), labels.view(-1))

        if return_value_only:
            return values
        else:
            # [0 0 0 0 prompt, answer, 0 0 0 0 ] for step 3, we have padding at the beginning
            # [prompt, answer, 0, 0, 0, 0] this is normal
            bs = values.size(0)
            seq_len = input_ids.shape[1]
            chosen_end_scores = [
            ]  # we use this name for consistency with the original forwad function
            for i in range(bs):
                input_id = input_ids[i]
                value = values[i]

                c_inds = (input_id == self.PAD_ID).nonzero()
                # here we only use the answer part of the sequence so we do not need to care about the padding at the beginning
                c_ind = c_inds[0].item() if len(
                    c_inds) > 0 else seq_len
                chosen_end_scores.append(value[c_ind - 1])
            return {
                "values": values,
                "chosen_end_scores": torch.stack(chosen_end_scores),
            }

    @torch.no_grad()
    def generate_instruction(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            **kwargs,

    ):

        kwargs["input_ids"] = input_ids
        kwargs["attention_mask"] = attention_mask
        kwargs["temperature"] = 1.0
        kwargs["max_length"] = self.logits_shape - 1

        output_actor = self.actor.generate(
            **kwargs
        )
        return output_actor

    @torch.no_grad()
    def generate(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            **kwargs,

    ):
        kwargs["input_ids"] = input_ids
        kwargs["attention_mask"] = attention_mask
        kwargs["max_length"] = self.logits_shape - 1

        output_critic = self.critic.generate(
            **kwargs
        )

        return output_critic
