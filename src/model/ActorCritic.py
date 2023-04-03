from dataclasses import dataclass
from typing import Tuple, Optional

import torch
from peft import get_peft_model, TaskType, LoraConfig
from torch.nn import CrossEntropyLoss,KLDivLoss
from transformers import AutoModelForSeq2SeqLM
from transformers.utils import ModelOutput


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

        self.actor = get_peft_model(actor, peft_config)

        critic = AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
        )
        self.critic = get_peft_model(critic, peft_config)

        self.actor_eps_clip = model_args.actor_eps_clip
        self.beta_s = model_args.beta_s
        self.pretrain_s = model_args.pretrain_s

        self.critic_eps_clip = model_args.actor_eps_clip
        self.logits_shape = model_args.logits_shape
        self.eps = 1e-8
        self.config =config


        sft = AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
        )
        self.sft = get_peft_model(sft, peft_config)


        for param in self.sft.parameters():
            param.requires_grad = False

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self.critic._shift_right(labels)

    @torch.no_grad()
    def getSFTlogits(self,
                     input_ids,attention_mask,
                     instruction_decoder_input_ids,
                     decoder_attention_mask,
                     instruction_labels,
                     use_cache,
                     output_attentions,
                     output_hidden_states,
                     return_dict):
        
        output_sft = self.sft(
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

        return output_sft

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
            old_rewards: Optional[torch.Tensor] = None,
            old_values: Optional[torch.Tensor] = None,
            old_actions_log_probs_mask: Optional[torch.Tensor] = None,
            instruction_labels : Optional[torch.Tensor] = None,
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


        loss = None
        rewards = None
        policy_loss = None
        kl_div_loss = None
        loss_lm_mean = None

        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100, reduction="none")
            loss_fct_mean = CrossEntropyLoss(ignore_index=-100)
            # kl_loss_mean = KLDivLoss(reduction="batchmean")

            loss_lm = loss_fct(output_critic.logits.view(-1, output_critic.logits.size(-1)), labels.view(-1))
            loss_lm_mean = loss_fct_mean(output_critic.logits.view(-1, output_critic.logits.size(-1)), labels.view(-1))
            
            # values = torch.exp(-loss_lm.clone())
            # Is there need detach()?

            rewards = torch.exp(-(loss_lm.view(output_critic.logits.shape[0], -1).sum(dim=-1) * (
                    (labels != -100).sum(dim=-1)))/10)


            if not self.training:
                return RLSeq2SeqLMOutput(
                loss=loss_lm_mean,
                prompt_logits=output_actor.logits,
                value_logits=rewards
            )

            output_stf = self.getSFTlogits(
                input_ids=input_ids,
                attention_mask=attention_mask,
                instruction_decoder_input_ids=instruction_decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                instruction_labels=instruction_labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            sft_logits = output_stf.logits

            sft_prob = torch.softmax(sft_logits,dim=-1)
            sft_log_prob = torch.log(sft_prob + self.eps)

            actions_logits = output_actor.logits

            action_pretrain_loss = loss_fct_mean(actions_logits.view(-1, actions_logits.size(-1)), instruction_labels.view(-1))
            # clean noise
            # actions_logits = actions_logits * (labels.unsqueeze(dim=-1)!=-100).float()
            # old_actions_log_probs = old_actions_log_probs * (labels.unsqueeze(dim=-1)!=-100).float()

            # get action log prob
            actions_prob_all = (
                torch.softmax(actions_logits, dim=-1)
            )
            actions_log_prob_all = torch.log(actions_prob_all + self.eps)


            actions_prob = actions_prob_all.max(dim=-1).values
            actions_log_prob = torch.log(actions_prob + self.eps)

            # clean noise
            action_mask = torch.bitwise_and(instruction_labels!=-100,old_actions_log_probs_mask.bool())
            # actions_log_prob = actions_log_prob * action_mask
            # old_actions_log_probs = old_actions_log_probs * action_mask

            # compute entropy
            entropies = ((actions_prob * actions_log_prob)*( action_mask.float())).sum(dim=-1)

            kl_div_loss = (
                ((actions_prob_all * (actions_log_prob_all - sft_log_prob))  * action_mask.unsqueeze(dim=-1).float())
                    .sum(dim=-1).sum(dim=-1)
                    .mean()
            )

            # compute PPO Loss -- Whan dimensions are different
            # (especially the values and the probs are
            #  multiplied directly with the reward)

            ## fix rations mask
            ratios = (actions_log_prob - old_actions_log_probs).exp() * action_mask.float()
            
            advantages = rewards.unsqueeze(dim=-1) - old_rewards
            surr1 = advantages * ratios

            # advantages = rewards.unsqueeze(dim=-1) - old_rewards
            # # normalize advantages
            # advantages = (advantages - advantages.mean(dim=-1)) / (
            #         advantages.std() + self.eps
            # )
            # surr1 = advantages * ratios
            surr2 = (
                    torch.clamp(ratios, 1 - self.actor_eps_clip, 1 + self.actor_eps_clip)
                    * advantages
            )

            policy_loss = -torch.min(surr1, surr2) - self.beta_s * entropies.unsqueeze(dim=-1)
            policy_loss = policy_loss.masked_select(action_mask).mean()
            loss = policy_loss + self.beta_s * kl_div_loss

            ## reward loss
            value_loss_clipped = old_rewards + (rewards - old_rewards).clamp(
                -self.critic_eps_clip, self.critic_eps_clip
            )
            value_loss1 = (value_loss_clipped - rewards) ** 2
            value_loss2 = (old_rewards - rewards) ** 2
            value_loss = torch.max(value_loss1, value_loss2).mean()

            loss += value_loss
            # loss += self.pretrain_s * (action_pretrain_loss + loss_lm_mean)

        return RLSeq2SeqLMOutput(
            loss=loss,
            prompt_logits=output_actor.logits,
            value_logits=rewards,
            policy_loss =policy_loss,
            kl_loss=kl_div_loss,
            critic_loss=value_loss
        )

    @torch.no_grad()
    def get_actor_logits(self,input_ids = None,attention_mask = None,decoder_input_ids = None,labels = None):
        if decoder_input_ids is None:
            decoder_input_ids = self.critic.prepare_decoder_input_ids_from_labels(labels)

        output = self.critic(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
        )

        actions_prob = (
            torch.softmax(output.logits, dim=-1).max(dim=-1).values
        )
        actions_log_prob = torch.log(actions_prob + self.eps)

        return actions_log_prob


    @torch.no_grad()
    def get_rewards(self,input_ids = None,attention_mask = None,decoder_input_ids = None,labels = None):

        if decoder_input_ids is None:
            decoder_input_ids = self.critic.prepare_decoder_input_ids_from_labels(labels)

        output_critic = self.critic(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
        )

        loss_fct = CrossEntropyLoss(ignore_index=-100, reduction="none")
        loss_lm = loss_fct(output_critic.logits.view(-1, output_critic.logits.size(-1)), labels.view(-1))
        rewards = torch.exp(-(loss_lm.clone().detach().view(output_critic.logits.shape[0], -1).sum(dim=-1) * (
                (labels != -100).sum(dim=-1)))/10)

        loss_lm = loss_lm.mean()

        return rewards


    @torch.no_grad()
    def generate_ppo(
            self,
            input_ids: torch.Tensor= None,
            attention_mask: torch.Tensor= None,
            instructed_input_ids = None,
            instructed_attention_mask = None,
            decoder_input_ids = None,
            labels = None,
            woinstruction_input_ids = None,
            woinstruction_mask = None,
            instruction_labels = None,
            instruction_decoder_input_ids = None,
            **kwargs,
    ):

        kwargs["input_ids"] = input_ids
        kwargs["attention_mask"] = attention_mask
        kwargs["output_scores"] = True
        kwargs["return_dict_in_generate"] = True
        kwargs["temperature"] = 0.9
        kwargs["max_length"] = self.logits_shape - 1 

        #             input_ids=input_ids,
        #             attention_mask=attention_mask,
        #             output_scores=True,
        #             return_dict_in_generate=True,

        # for peft
        output_actor = self.actor.generate(
            **kwargs
        )

        rewards = self.get_rewards(instructed_input_ids,instructed_attention_mask,decoder_input_ids=decoder_input_ids,labels=labels)
        return output_actor.sequences,rewards

    @torch.no_grad()
    def generate_instruction(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            **kwargs,

    ):
        """Generate actions, actions_logits, values and sequences from states

        Args:
            states (torch.Tensor): user inputs
            state_mask (torch.Tensor): Mask for the states of the environment

        Returns:
            actions (torch.Tensor): Actions generated from the states
            actions_logits (torch.Tensor): Logits for the actions generated
                from the states (i.e. pi(y | x))
            values (torch.Tensor): Values generated by the critic model
                for the actions generated by the actor (i.e. V(x))
            sequences (torch.Tensor): Sequences generated from the states
                as [states, actions]
        """
        kwargs["input_ids"] = input_ids
        kwargs["attention_mask"] = attention_mask
        kwargs["temperature"] = 0.9
        kwargs["max_length"] =  self.logits_shape - 1 

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
        """Generate actions, actions_logits, values and sequences from states

        Args:
            states (torch.Tensor): user inputs
            state_mask (torch.Tensor): Mask for the states of the environment

        Returns:
            actions (torch.Tensor): Actions generated from the states
            actions_logits (torch.Tensor): Logits for the actions generated
                from the states (i.e. pi(y | x))
            values (torch.Tensor): Values generated by the critic model
                for the actions generated by the actor (i.e. V(x))
            sequences (torch.Tensor): Sequences generated from the states
                as [states, actions]
        """
        kwargs["input_ids"] = input_ids
        kwargs["attention_mask"] = attention_mask
        kwargs["max_length"] =  self.logits_shape - 1 

        output_critic = self.critic.generate(
            **kwargs
        )

        return output_critic

        # # generate prompts -> actor
        # # generate answer -> critic
        # pass
        # # actions, sequence = self.actor.generate(input_ids, attention_mask)
        # # sequences_mask = sequence != self.actor.tokenizer.pad_token_id
        # # sequences_mask = sequences_mask.to(sequence.device).long().detach()
        # #
        # # # generate actions_logits and values
        # # actions_logits, values = self.critic.forward(
        # #     sequence, sequences_mask, labels
        # # )
