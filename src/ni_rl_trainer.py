import string
import re
from typing import Deque

import torch
from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.utils import is_accelerate_available
from transformers.trainer import *
from datasets import load_metric
from transformers.trainer_callback import TrainerCallback
from collections import deque, namedtuple
from datasets import Dataset
import jsonlines
skip_first_batches = None
if is_accelerate_available():
    from accelerate import __version__ as accelerate_version

    if version.parse(accelerate_version) >= version.parse("0.16"):
        from accelerate import skip_first_batches

class DenserEvalCallback(TrainerCallback):

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):

        log_eval_steps = [1, 50, 100, 200]

        # Log
        if args.logging_strategy == IntervalStrategy.STEPS and state.global_step in log_eval_steps:
            control.should_log = True

        # Evaluate
        if args.evaluation_strategy == IntervalStrategy.STEPS and state.global_step in log_eval_steps:
            control.should_evaluate = True

        return control




# class ExperienceDataset(Dataset):
#     """Dataset to train the actor-critic models"""

#     def __init__(
#             self,
#             memories: Deque[Memory]
#     ) -> None:
#         super().__init__()
#         self.data = list(memories)
#         self.len = len(self.data)
        
#     def __len__(
#             self,
#     ) -> int:
#         return self.len

#     def __getitem__(self, idx):
#         # return the idx-th memory element as a tuple of tensors on the device
#         item = {
#             "instances":self.data[idx].instances,
#             "instructions":self.data[idx].instructions,
#             "items":self.data[idx].items,
#             "labels":self.data[idx].labels,
#             "reward":self.data[idx].rewards,
#             "logits":self.data[idx].logits,
#             "logits_mask":self.data[idx].logits_mask,
#         }

#         return item


class NIRLTrainer(Seq2SeqTrainer):

    def __init__(self, model: Union[PreTrainedModel, nn.Module] = None,
                 args: TrainingArguments = None,
                 data_collator: Optional[DataCollator] = None,
                 train_dataset: Optional[Dataset] = None,
                 eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
                 tokenizer: Optional[PreTrainedTokenizerBase] = None,
                 model_init: Optional[Callable[[], PreTrainedModel]] = None,
                 compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
                 callbacks: Optional[List[TrainerCallback]] = None,
                 optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
                 preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
                 ex_train_dataset: Optional[Dataset] = None,
                 ex_data_collator: Optional[DataCollator] = None,
                 ):

        super().__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer, model_init,
                         compute_metrics, callbacks, optimizers, preprocess_logits_for_metrics)
        self.ex_train_dataset = ex_train_dataset
        self.ex_data_collator = ex_data_collator
        self.current_episode = 0 
        self.ppo_beam = args.ppo_beam
    # rewrite the evaluation loop, with customized call to compute_metrics
    def _get_ex_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.ex_train_dataset is None or not has_length(self.ex_train_dataset):
            return None

        generator = None
        if self.args.world_size <= 1:
            generator = torch.Generator()
            # for backwards compatibility, we generate a seed here (which is sampled from a generator seeded with
            # `args.seed`) if data_seed isn't provided.
            # Further on in this method, we default to `args.seed` instead.
            if self.args.data_seed is None:
                seed = int(torch.empty((), dtype=torch.int64).random_().item())
            else:
                seed = self.args.data_seed
            generator.manual_seed(seed)

        seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed

        # Build the sampler.
        if self.args.group_by_length:
            if is_datasets_available() and isinstance(self.ex_train_dataset, datasets.Dataset):
                lengths = (
                    self.ex_train_dataset[self.args.length_column_name]
                    if self.args.length_column_name in self.ex_train_dataset.column_names
                    else None
                )
            else:
                lengths = None
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
            if self.args.world_size <= 1:
                return LengthGroupedSampler(
                    self.args.train_batch_size * self.args.gradient_accumulation_steps ,
                    dataset=self.ex_train_dataset,
                    lengths=lengths,
                    model_input_name=model_input_name,
                    generator=generator,
                )
            else:
                return DistributedLengthGroupedSampler(
                    self.args.train_batch_size * self.args.gradient_accumulation_steps,
                    dataset=self.ex_train_dataset,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    lengths=lengths,
                    model_input_name=model_input_name,
                    seed=seed,
                )

        else:
            if self.args.world_size <= 1:
                return RandomSampler(self.ex_train_dataset, generator=generator)
            elif (
                    self.args.parallel_mode in [ParallelMode.TPU, ParallelMode.SAGEMAKER_MODEL_PARALLEL]
                    and not self.args.dataloader_drop_last
            ):
                # Use a loop for TPUs when drop_last is False to have all batches have the same size.
                return DistributedSamplerWithLoop(
                    self.ex_train_dataset,
                    batch_size=self.args.per_device_train_batch_size,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    seed=seed,
                )
            else:
                return DistributedSampler(
                    self.ex_train_dataset,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    seed=seed,
                )

    def get_ex_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.ex_train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.ex_train_dataset
        data_collator = self.ex_data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        if isinstance(train_dataset, torch.utils.data.IterableDataset):
            if self.args.world_size > 1:
                train_dataset = IterableDatasetShard(
                    train_dataset,
                    batch_size=self._train_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )

            return DataLoader(
                train_dataset,
                batch_size=self._train_batch_size,
                collate_fn=data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        train_sampler = self._get_ex_train_sampler()

        return DataLoader(
            train_dataset,
            batch_size=self._train_batch_size,
            sampler=train_sampler,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            worker_init_fn=seed_worker,
        )


    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        generator = None
        if self.args.world_size <= 1:
            generator = torch.Generator()
            # for backwards compatibility, we generate a seed here (which is sampled from a generator seeded with
            # `args.seed`) if data_seed isn't provided.
            # Further on in this method, we default to `args.seed` instead.
            if self.args.data_seed is None:
                seed = int(torch.empty((), dtype=torch.int64).random_().item())
            else:
                seed = self.args.data_seed
            generator.manual_seed(seed)

        seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed

        # Build the sampler.
        if self.args.group_by_length:
            if is_datasets_available() and isinstance(self.train_dataset, datasets.Dataset):
                lengths = (
                    self.train_dataset[self.args.length_column_name]
                    if self.args.length_column_name in self.train_dataset.column_names
                    else None
                )
            else:
                lengths = None
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
            if self.args.world_size <= 1:
                return LengthGroupedSampler(
                    self.args.train_batch_size * self.args.gradient_accumulation_steps * 4 ,
                    dataset=self.train_dataset,
                    lengths=lengths,
                    model_input_name=model_input_name,
                    generator=generator,
                )
            else:
                return DistributedLengthGroupedSampler(
                    self.args.train_batch_size * self.args.gradient_accumulation_steps * 4,
                    dataset=self.train_dataset,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    lengths=lengths,
                    model_input_name=model_input_name,
                    seed=seed,
                )

        else:
            if self.args.world_size <= 1:
                return RandomSampler(self.train_dataset, generator=generator)
            elif (
                self.args.parallel_mode in [ParallelMode.TPU, ParallelMode.SAGEMAKER_MODEL_PARALLEL]
                and not self.args.dataloader_drop_last
            ):
                # Use a loop for TPUs when drop_last is False to have all batches have the same size.
                return DistributedSamplerWithLoop(
                    self.train_dataset,
                    batch_size=self.args.per_device_train_batch_size * 4,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    seed=seed,
                )
            else:
                return DistributedSampler(
                    self.train_dataset,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    seed=seed,
                )

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        if isinstance(train_dataset, torch.utils.data.IterableDataset):
            if self.args.world_size > 1:
                train_dataset = IterableDatasetShard(
                    train_dataset,
                    batch_size=self._train_batch_size * 4,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )

            return DataLoader(
                train_dataset,
                batch_size=self.args.per_device_train_batch_size * 4 ,
                collate_fn=data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        train_sampler = self._get_train_sampler()

        return DataLoader(
            train_dataset,
            batch_size=self._train_batch_size * 4,
            sampler=train_sampler,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            worker_init_fn=seed_worker,
        )


    def _inner_training_loop(
            self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):

        train_dataloader = self.get_train_dataloader()
        # if eval is called w/o train init deepspeed here
        if args.deepspeed and not self.deepspeed:
            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(
                self, num_training_steps=0, resume_from_checkpoint=None
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine

        model = self._wrap_model(self.model, training=False, dataloader=train_dataloader)

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        for i in range(args.episode):

            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(i)
            elif hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDatasetShard):
                train_dataloader.dataset.set_epoch(i)

            logger.info("***** Running RL training *****")
            logger.info(f"  Num Episode = {i}")
            logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")

            generated_data = self.ppo_generate_data(model, train_dataloader,i)
            # generated_data = [x._asdict() for x in list(generated_data)]

            def dataset_translator():
                for i in range(len(generated_data)):
                    yield generated_data[i]

            self.ex_train_dataset = Dataset.from_generator(dataset_translator)
            output= self._inner_mini_training_loop(batch_size, args, resume_from_checkpoint, trial, ignore_keys_for_eval)
            self.current_episode += 1 
        return output
    
    # def _inner_mini_training_loop(
    #         self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    # ):
    #     self._train_batch_size = batch_size
    #     # Data loader and number of training steps
    #     train_dataloader = self.get_ex_train_dataloader()
    #
    #     # Setting up training control variables:
    #     # number of training epochs: num_train_epochs
    #     # number of training steps per epoch: num_update_steps_per_epoch
    #     # total number of training steps to execute: max_steps
    #     total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size
    #
    #     len_dataloader = None
    #     if has_length(train_dataloader):
    #         len_dataloader = len(train_dataloader)
    #         num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
    #         num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
    #         num_examples = self.num_examples(train_dataloader)
    #         if args.max_steps > 0:
    #             max_steps = args.max_steps
    #             num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
    #                 args.max_steps % num_update_steps_per_epoch > 0
    #             )
    #             # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
    #             # the best we can do.
    #             num_train_samples = args.max_steps * total_train_batch_size
    #         else:
    #             max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
    #             num_train_epochs = math.ceil(args.num_train_epochs)
    #             num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
    #     elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
    #         max_steps = args.max_steps
    #         # Setting a very large number of epochs so we go as many times as necessary over the iterator.
    #         num_train_epochs = sys.maxsize
    #         num_update_steps_per_epoch = max_steps
    #         num_examples = total_train_batch_size * args.max_steps
    #         num_train_samples = args.max_steps * total_train_batch_size
    #     else:
    #         raise ValueError(
    #             "args.max_steps must be set to a positive value if dataloader does not have a length, was"
    #             f" {args.max_steps}"
    #         )
    #
    #     if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
    #         if self.args.n_gpu > 1:
    #             # nn.DataParallel(model) replicates the model, creating new variables and module
    #             # references registered here no longer work on other gpus, breaking the module
    #             raise ValueError(
    #                 "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
    #                 " (torch.distributed.launch)."
    #             )
    #         else:
    #             debug_overflow = DebugUnderflowOverflow(self.model)  # noqa
    #
    #     delay_optimizer_creation = (
    #             self.sharded_ddp is not None
    #             and self.sharded_ddp != ShardedDDPOption.SIMPLE
    #             or is_sagemaker_mp_enabled()
    #             or self.fsdp is not None
    #     )
    #     if args.deepspeed:
    #         deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
    #             self, num_training_steps=max_steps, resume_from_checkpoint=resume_from_checkpoint
    #         )
    #         self.model = deepspeed_engine.module
    #         self.model_wrapped = deepspeed_engine
    #         self.deepspeed = deepspeed_engine
    #         self.optimizer = optimizer
    #         self.lr_scheduler = lr_scheduler
    #     elif not delay_optimizer_creation:
    #         self.create_optimizer_and_scheduler(num_training_steps=max_steps)
    #
    #     self.state = TrainerState()
    #     self.state.is_hyper_param_search = trial is not None
    #
    #     # Activate gradient checkpointing if needed
    #     if args.gradient_checkpointing:
    #         self.model.gradient_checkpointing_enable()
    #
    #     model = self._wrap_model(self.model_wrapped)
    #
    #     if is_sagemaker_mp_enabled() and resume_from_checkpoint is not None:
    #         self._load_from_checkpoint(resume_from_checkpoint, model)
    #
    #     # for the rest of this function `model` is the outside model, whether it was wrapped or not
    #     if model is not self.model:
    #         self.model_wrapped = model
    #
    #     if delay_optimizer_creation:
    #         self.create_optimizer_and_scheduler(num_training_steps=max_steps)
    #
    #     # Check if saved optimizer or scheduler states exist
    #     self._load_optimizer_and_scheduler(resume_from_checkpoint)
    #
    #     # important: at this point:
    #     # self.model         is the Transformers Model
    #     # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.
    #
    #     # Train!
    #     logger.info("***** Running training *****")
    #     logger.info(f"  Num examples = {num_examples}")
    #     logger.info(f"  Num Epochs = {num_train_epochs}")
    #     logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    #     logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
    #     logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    #     logger.info(f"  Total optimization steps = {max_steps}")
    #     logger.info(
    #         f"  Number of trainable parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    #     )
    #
    #     self.state.epoch = 0
    #     start_time = time.time()
    #     epochs_trained = 0
    #     steps_trained_in_current_epoch = 0
    #     steps_trained_progress_bar = None
    #
    #     # Check if continuing training from a checkpoint
    #     if resume_from_checkpoint is not None and os.path.isfile(
    #             os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
    #     ):
    #         self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
    #         epochs_trained = self.state.global_step // num_update_steps_per_epoch
    #         if not args.ignore_data_skip:
    #             steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
    #             steps_trained_in_current_epoch *= args.gradient_accumulation_steps
    #         else:
    #             steps_trained_in_current_epoch = 0
    #
    #         logger.info("  Continuing training from checkpoint, will skip to saved global_step")
    #         logger.info(f"  Continuing training from epoch {epochs_trained}")
    #         logger.info(f"  Continuing training from global step {self.state.global_step}")
    #         if not args.ignore_data_skip:
    #             if skip_first_batches is None:
    #                 logger.info(
    #                     f"  Will skip the first {epochs_trained} epochs then the first"
    #                     f" {steps_trained_in_current_epoch} batches in the first epoch. If this takes a lot of time,"
    #                     " you can install the latest version of Accelerate with `pip install -U accelerate`.You can"
    #                     " also add the `--ignore_data_skip` flag to your launch command, but you will resume the"
    #                     " training on data already seen by your model."
    #                 )
    #             else:
    #                 logger.info(
    #                     f"  Will skip the first {epochs_trained} epochs then the first"
    #                     f" {steps_trained_in_current_epoch} batches in the first epoch."
    #                 )
    #             if self.is_local_process_zero() and not args.disable_tqdm and skip_first_batches is None:
    #                 steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
    #                 steps_trained_progress_bar.set_description("Skipping the first batches")
    #
    #     # Update the references
    #     self.callback_handler.model = self.model
    #     self.callback_handler.optimizer = self.optimizer
    #     self.callback_handler.lr_scheduler = self.lr_scheduler
    #     self.callback_handler.train_dataloader = train_dataloader
    #     if self.hp_name is not None and self._trial is not None:
    #         # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
    #         # parameter to Train when using DDP.
    #         self.state.trial_name = self.hp_name(self._trial)
    #     if trial is not None:
    #         assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
    #         self.state.trial_params = hp_params(assignments)
    #     else:
    #         self.state.trial_params = None
    #     # This should be the same if the state has been saved but in case the training arguments changed, it's safer
    #     # to set this after the load.
    #     self.state.max_steps = max_steps
    #     self.state.num_train_epochs = num_train_epochs
    #     self.state.is_local_process_zero = self.is_local_process_zero()
    #     self.state.is_world_process_zero = self.is_world_process_zero()
    #
    #     # tr_loss is a tensor to avoid synchronization of TPUs through .item()
    #     tr_loss = torch.tensor(0.0).to(args.device)
    #     # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
    #     self._total_loss_scalar = 0.0
    #     self._globalstep_last_logged = self.state.global_step
    #     model.zero_grad()
    #
    #     self.control = self.callback_handler.on_train_begin(args, self.state, self.control)
    #
    #     # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
    #     if not args.ignore_data_skip:
    #         for epoch in range(epochs_trained):
    #             is_random_sampler = hasattr(train_dataloader, "sampler") and isinstance(
    #                 train_dataloader.sampler, RandomSampler
    #             )
    #             if is_torch_less_than_1_11 or not is_random_sampler:
    #                 # We just need to begin an iteration to create the randomization of the sampler.
    #                 # That was before PyTorch 1.11 however...
    #                 for _ in train_dataloader:
    #                     break
    #             else:
    #                 # Otherwise we need to call the whooooole sampler cause there is some random operation added
    #                 # AT THE VERY END!
    #                 _ = list(train_dataloader.sampler)
    #
    #     total_batched_samples = 0
    #     for epoch in range(epochs_trained, num_train_epochs):
    #         if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
    #             train_dataloader.sampler.set_epoch(epoch)
    #         elif hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDatasetShard):
    #             train_dataloader.dataset.set_epoch(epoch)
    #
    #         if is_torch_tpu_available():
    #             parallel_loader = pl.ParallelLoader(train_dataloader, [args.device]).per_device_loader(args.device)
    #             epoch_iterator = parallel_loader
    #         else:
    #             epoch_iterator = train_dataloader
    #
    #         # Reset the past mems state at the beginning of each epoch if necessary.
    #         if args.past_index >= 0:
    #             self._past = None
    #
    #         steps_in_epoch = (
    #             len(epoch_iterator)
    #             if len_dataloader is not None
    #             else args.max_steps * args.gradient_accumulation_steps
    #         )
    #         self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)
    #
    #         if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
    #             self._load_rng_state(resume_from_checkpoint)
    #
    #         rng_to_sync = False
    #         steps_skipped = 0
    #         if skip_first_batches is not None and steps_trained_in_current_epoch > 0:
    #             epoch_iterator = skip_first_batches(epoch_iterator, steps_trained_in_current_epoch)
    #             steps_skipped = steps_trained_in_current_epoch
    #             steps_trained_in_current_epoch = 0
    #             rng_to_sync = True
    #
    #         step = -1
    #         for step, inputs in enumerate(epoch_iterator):
    #             total_batched_samples += 1
    #             if rng_to_sync:
    #                 self._load_rng_state(resume_from_checkpoint)
    #                 rng_to_sync = False
    #
    #             # Skip past any already trained steps if resuming training
    #             if steps_trained_in_current_epoch > 0:
    #                 steps_trained_in_current_epoch -= 1
    #                 if steps_trained_progress_bar is not None:
    #                     steps_trained_progress_bar.update(1)
    #                 if steps_trained_in_current_epoch == 0:
    #                     self._load_rng_state(resume_from_checkpoint)
    #                 continue
    #             elif steps_trained_progress_bar is not None:
    #                 steps_trained_progress_bar.close()
    #                 steps_trained_progress_bar = None
    #
    #             if step % args.gradient_accumulation_steps == 0:
    #                 self.control = self.callback_handler.on_step_begin(args, self.state, self.control)
    #
    #             if (
    #                     (total_batched_samples % args.gradient_accumulation_steps != 0)
    #                     and args.local_rank != -1
    #                     and args._no_sync_in_gradient_accumulation
    #             ):
    #                 # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
    #                 with model.no_sync():
    #                     tr_loss_step = self.training_step(model, inputs)
    #             else:
    #                 tr_loss_step = self.training_step(model, inputs)
    #
    #             if (
    #                     args.logging_nan_inf_filter
    #                     and not is_torch_tpu_available()
    #                     and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
    #             ):
    #                 # if loss is nan or inf simply add the average of previous logged losses
    #                 tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
    #             else:
    #                 tr_loss += tr_loss_step
    #
    #             self.current_flos += float(self.floating_point_ops(inputs))
    #
    #             # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
    #             if self.deepspeed:
    #                 self.deepspeed.step()
    #
    #             if total_batched_samples % args.gradient_accumulation_steps == 0 or (
    #                     # last step in epoch but step is always smaller than gradient_accumulation_steps
    #                     steps_in_epoch <= args.gradient_accumulation_steps
    #                     and (step + 1) == steps_in_epoch
    #             ):
    #                 # Gradient clipping
    #                 if args.max_grad_norm is not None and args.max_grad_norm > 0 and not self.deepspeed:
    #                     # deepspeed does its own clipping
    #
    #                     if self.do_grad_scaling:
    #                         # Reduce gradients first for XLA
    #                         if is_torch_tpu_available():
    #                             gradients = xm._fetch_gradients(self.optimizer)
    #                             xm.all_reduce("sum", gradients, scale=1.0 / xm.xrt_world_size())
    #                         # AMP: gradients need unscaling
    #                         self.scaler.unscale_(self.optimizer)
    #
    #                     if is_sagemaker_mp_enabled() and args.fp16:
    #                         self.optimizer.clip_master_grads(args.max_grad_norm)
    #                     elif hasattr(self.optimizer, "clip_grad_norm"):
    #                         # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
    #                         self.optimizer.clip_grad_norm(args.max_grad_norm)
    #                     elif hasattr(model, "clip_grad_norm_"):
    #                         # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
    #                         model.clip_grad_norm_(args.max_grad_norm)
    #                     else:
    #                         # Revert to normal clipping otherwise, handling Apex or full precision
    #                         nn.utils.clip_grad_norm_(
    #                             amp.master_params(self.optimizer) if self.use_apex else model.parameters(),
    #                             args.max_grad_norm,
    #                         )
    #
    #                 # Optimizer step
    #                 optimizer_was_run = True
    #                 if self.deepspeed:
    #                     pass  # called outside the loop
    #                 elif is_torch_tpu_available():
    #                     if self.do_grad_scaling:
    #                         self.scaler.step(self.optimizer)
    #                         self.scaler.update()
    #                     else:
    #                         xm.optimizer_step(self.optimizer)
    #                 elif self.do_grad_scaling:
    #                     scale_before = self.scaler.get_scale()
    #                     self.scaler.step(self.optimizer)
    #                     self.scaler.update()
    #                     scale_after = self.scaler.get_scale()
    #                     optimizer_was_run = scale_before <= scale_after
    #                 else:
    #                     self.optimizer.step()
    #
    #                 if optimizer_was_run and not self.deepspeed:
    #                     self.lr_scheduler.step()
    #
    #                 model.zero_grad()
    #                 self.state.global_step += 1
    #                 self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
    #                 self.control = self.callback_handler.on_step_end(args, self.state, self.control)
    #
    #                 self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
    #             else:
    #                 self.control = self.callback_handler.on_substep_end(args, self.state, self.control)
    #
    #             if self.control.should_epoch_stop or self.control.should_training_stop:
    #                 break
    #         if step < 0:
    #             logger.warning(
    #                 "There seems to be not a single sample in your epoch_iterator, stopping training at step"
    #                 f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
    #                 f" num_steps ({max_steps}) higher than the number of available samples."
    #             )
    #             self.control.should_training_stop = True
    #
    #         self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
    #         self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
    #
    #         if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
    #             if is_torch_tpu_available():
    #                 # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
    #                 xm.master_print(met.metrics_report())
    #             else:
    #                 logger.warning(
    #                     "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
    #                     "configured. Check your training configuration if this is unexpected."
    #                 )
    #         if self.control.should_training_stop:
    #             break
    #
    #     if args.past_index and hasattr(self, "_past"):
    #         # Clean the state at the end of training
    #         delattr(self, "_past")
    #
    #     logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
    #     if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
    #         # Wait for everyone to get here so we are sur the model has been saved by process 0.
    #         if is_torch_tpu_available():
    #             xm.rendezvous("load_best_model_at_end")
    #         elif args.local_rank != -1:
    #             dist.barrier()
    #         elif is_sagemaker_mp_enabled():
    #             smp.barrier()
    #
    #         self._load_best_model()
    #
    #     # add remaining tr_loss
    #     self._total_loss_scalar += tr_loss.item()
    #     train_loss = self._total_loss_scalar / self.state.global_step
    #
    #     metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
    #     self.store_flos()
    #     metrics["total_flos"] = self.state.total_flos
    #     metrics["train_loss"] = train_loss
    #
    #     self.is_in_train = False
    #
    #     self._memory_tracker.stop_and_update_metrics(metrics)
    #
    #     self.log(metrics)
    #
    #     run_dir = self._get_output_dir(trial)
    #     checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)
    #
    #     # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
    #     if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
    #         for checkpoint in checkpoints_sorted:
    #             if checkpoint != self.state.best_model_checkpoint:
    #                 logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
    #                 shutil.rmtree(checkpoint)
    #
    #     self.control = self.callback_handler.on_train_end(args, self.state, self.control)
    #
    #     return TrainOutput(self.state.global_step, train_loss, metrics)

    def _inner_mini_training_loop(
            self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self._train_batch_size = batch_size
        # Data loader and number of training steps
        train_dataloader = self.get_ex_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                f"args.max_steps must be set to a positive value if dataloader does not have a length, was {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP (torch.distributed.launch)."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = (
                self.sharded_ddp is not None
                and self.sharded_ddp != ShardedDDPOption.SIMPLE
                or is_sagemaker_mp_enabled()
                or self.fsdp is not None
        )
        if args.deepspeed:
            deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
                self, num_training_steps=max_steps, resume_from_checkpoint=resume_from_checkpoint
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
        elif not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch. If this takes a lot of time, you can add the `--ignore_data_skip` "
                    "flag to your launch command, but you will resume the training on data already seen by your model."
                )
                if self.is_local_process_zero() and not args.disable_tqdm:
                    steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
                    steps_trained_progress_bar.set_description("Skipping the first batches")

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        self.state.trial_name = self.hp_name(trial) if self.hp_name is not None else None
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        mlm_loss = torch.tensor(0.0).to(args.device)
        bce_loss = torch.tensor(0.0).to(args.device)
        bce_acc = torch.tensor(0.0).to(args.device)

        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._total_mlm_loss_scalar = 0.0
        self._total_bce_loss_scalar = 0.0
        self._total_bce_acc_scalar = 0.0

        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                is_random_sampler = hasattr(train_dataloader, "sampler") and isinstance(
                    train_dataloader.sampler, RandomSampler
                )
                if version.parse(torch.__version__) < version.parse("1.11") or not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    # That was before PyTorch 1.11 however...
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    _ = list(train_dataloader.sampler)

        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            elif hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDatasetShard):
                train_dataloader.dataset.set_epoch(epoch)

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [args.device]).per_device_loader(args.device)
                epoch_iterator = parallel_loader
            else:
                epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            step = -1
            for step, inputs in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                if (
                        ((step + 1) % args.gradient_accumulation_steps != 0)
                        and args.local_rank != -1
                        and args._no_sync_in_gradient_accumulation
                ):
                    # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                    with model.no_sync():
                        tr_loss_step, mlm_loss_step, bce_loss_step, bce_acc_step = self.training_step(model, inputs)
                else:
                    tr_loss_step, mlm_loss_step, bce_loss_step, bce_acc_step = self.training_step(model, inputs)

                if (
                        args.logging_nan_inf_filter
                        and not is_torch_tpu_available()
                        and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                    mlm_loss += mlm_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                    bce_loss += bce_loss_step / (1 + self.state.global_step - self._globalstep_last_logged)
                    bce_acc += bce_acc_step / (1 + self.state.global_step - self._globalstep_last_logged)

                else:
                    tr_loss += tr_loss_step
                    mlm_loss += mlm_loss_step
                    bce_loss += bce_loss_step
                    bce_acc += bce_acc_step

                self.current_flos += float(self.floating_point_ops(inputs))

                # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
                if self.deepspeed:
                    self.deepspeed.step()

                if (step + 1) % args.gradient_accumulation_steps == 0 or (
                        # last step in epoch but step is always smaller than gradient_accumulation_steps
                        steps_in_epoch <= args.gradient_accumulation_steps
                        and (step + 1) == steps_in_epoch
                ):
                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0 and not self.deepspeed:
                        # deepspeed does its own clipping

                        if self.do_grad_scaling:
                            # Reduce gradients first for XLA
                            if is_torch_tpu_available():
                                gradients = xm._fetch_gradients(self.optimizer)
                                xm.all_reduce("sum", gradients, scale=1.0 / xm.xrt_world_size())
                            # AMP: gradients need unscaling
                            self.scaler.unscale_(self.optimizer)

                        if hasattr(self.optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            self.optimizer.clip_grad_norm(args.max_grad_norm)
                        elif hasattr(model, "clip_grad_norm_"):
                            # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                            model.clip_grad_norm_(args.max_grad_norm)
                        else:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer) if self.use_apex else model.parameters(),
                                args.max_grad_norm,
                            )

                    # Optimizer step
                    optimizer_was_run = True
                    if self.deepspeed:
                        pass  # called outside the loop
                    elif is_torch_tpu_available():
                        if self.do_grad_scaling:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            xm.optimizer_step(self.optimizer)
                    elif self.do_grad_scaling:
                        scale_before = self.scaler.get_scale()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        scale_after = self.scaler.get_scale()
                        optimizer_was_run = scale_before <= scale_after
                    else:
                        self.optimizer.step()

                    if optimizer_was_run and not self.deepspeed:
                        self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, mlm_loss, bce_loss,bce_acc, model, trial, epoch,
                                                  ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
            if step < 0:
                logger.warning(
                    f"There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, mlm_loss, bce_loss,bce_acc, model, trial, epoch, ignore_keys_for_eval)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.local_rank != -1:
                dist.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        self._total_mlm_loss_scalar += mlm_loss.item()
        self._total_bce_loss_scalar += bce_loss.item()
        self._total_bce_acc_scalar += bce_acc.item()

        train_loss = self._total_loss_scalar / self.state.global_step
        train_mlm_loss = self._total_mlm_loss_scalar / self.state.global_step
        train_bce_loss = self._total_bce_loss_scalar / self.state.global_step
        train_bce_acc = self._total_bce_acc_scalar / self.state.global_step

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss
        metrics["policy_loss"] = train_mlm_loss
        metrics["kl_loss"] = train_bce_loss
        metrics["critic_loss"] = train_bce_acc


        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        return TrainOutput(self.state.global_step, train_loss, metrics)

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            scaler = self.scaler if self.do_grad_scaling else None
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps, scaler=scaler)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.autocast_smart_context_manager():
            loss, mlm_loss, bce_loss, bce_acc = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
            mlm_loss = mlm_loss.mean()  # mean() to average on multi-gpu parallel training
            bce_loss = bce_loss.mean()  # mean() to average on multi-gpu parallel training
            bce_acc = bce_acc.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps
            mlm_loss = mlm_loss / self.args.gradient_accumulation_steps
            bce_loss = bce_loss / self.args.gradient_accumulation_steps
            bce_acc = bce_acc / self.args.gradient_accumulation_steps

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()

        if self.model.training:
            return loss.detach(), mlm_loss.detach(), bce_loss.detach(), bce_acc.detach()
        else:
            return loss.detach()

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        if self.model.training:
            return (loss, outputs) if return_outputs else loss, outputs['policy_loss'], outputs['kl_loss'], outputs['critic_loss']
        else:
            return (loss, outputs) if return_outputs else loss


    def _maybe_log_save_evaluate(self, tr_loss, mlm_loss, bce_loss, bce_acc, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:
            if is_torch_tpu_available():
                xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()
            mlm_loss_loss_scalar = self._nested_gather(mlm_loss).mean().item()
            bce_loss_loss_scalar = self._nested_gather(bce_loss).mean().item()
            bce_acc_loss_scalar = self._nested_gather(bce_acc).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss
            mlm_loss -= mlm_loss
            bce_loss -= bce_loss
            bce_acc -= bce_acc

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["policy_loss"] = round(mlm_loss_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["kl_loss"] = round(bce_loss_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["critic_loss"] = round(bce_acc_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)


            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._total_mlm_loss_scalar += mlm_loss_loss_scalar
            self._total_bce_loss_scalar += bce_loss_loss_scalar
            self._total_bce_acc_scalar += bce_acc_loss_scalar

            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, epoch, metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def _save_checkpoint(self, model, trial, metrics=None):
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        # Save model checkpoint
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}-{int(self.current_episode/20)}"

        if self.hp_search_backend is None and trial is None:
            self.store_flos()

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        self.save_model(output_dir, _internal_call=True)
        if self.deepspeed:
            # under zero3 model file itself doesn't get saved since it's bogus! Unless deepspeed
            # config `stage3_gather_16bit_weights_on_model_save` is True
            self.deepspeed.save_checkpoint(output_dir)

        # Save optimizer and scheduler
        if self.sharded_ddp == ShardedDDPOption.SIMPLE:
            self.optimizer.consolidate_state_dict()

        if is_torch_tpu_available():
            xm.rendezvous("saving_optimizer_states")
            xm.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))
            with warnings.catch_warnings(record=True) as caught_warnings:
                xm.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
                reissue_pt_warnings(caught_warnings)
        elif is_sagemaker_mp_enabled():
            opt_state_dict = self.optimizer.local_state_dict(gather_if_shard=False)
            smp.barrier()
            if smp.rdp_rank() == 0 or smp.state.cfg.shard_optimizer_state:
                smp.save(
                    opt_state_dict,
                    os.path.join(output_dir, OPTIMIZER_NAME),
                    partial=True,
                    v3=smp.state.cfg.shard_optimizer_state,
                )
            if self.args.should_save:
                with warnings.catch_warnings(record=True) as caught_warnings:
                    torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
                reissue_pt_warnings(caught_warnings)
                if self.do_grad_scaling:
                    torch.save(self.scaler.state_dict(), os.path.join(output_dir, SCALER_NAME))
        elif self.args.should_save and not self.deepspeed:
            # deepspeed.save_checkpoint above saves model/optim/sched
            torch.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))
            with warnings.catch_warnings(record=True) as caught_warnings:
                torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
            reissue_pt_warnings(caught_warnings)
            if self.do_grad_scaling:
                torch.save(self.scaler.state_dict(), os.path.join(output_dir, SCALER_NAME))

        # Determine the new best metric / best model checkpoint
        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]

            operator = np.greater if self.args.greater_is_better else np.less
            if (
                self.state.best_metric is None
                or self.state.best_model_checkpoint is None
                or operator(metric_value, self.state.best_metric)
            ):
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir

        # Save the Trainer state
        if self.args.should_save:
            self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

        # Save RNG state in non-distributed training
        rng_states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cpu": torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            if self.args.local_rank == -1:
                # In non distributed, we save the global CUDA RNG state (will take care of DataParallel)
                rng_states["cuda"] = torch.cuda.random.get_rng_state_all()
            else:
                rng_states["cuda"] = torch.cuda.random.get_rng_state()

        if is_torch_tpu_available():
            rng_states["xla"] = xm.get_rng_state()

        # A process can arrive here before the process 0 has a chance to save the model, in which case output_dir may
        # not yet exist.
        os.makedirs(output_dir, exist_ok=True)

        if self.args.world_size <= 1:
            torch.save(rng_states, os.path.join(output_dir, "rng_state.pth"))
        else:
            torch.save(rng_states, os.path.join(output_dir, f"rng_state_{self.args.process_index}.pth"))

        if self.args.push_to_hub:
            self._push_from_checkpoint(output_dir)

        # Maybe delete some older checkpoints.
        if self.args.should_save:
            self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)


    @torch.no_grad()
    def ppo_generate_data(self, model, epoch_iterator,episode):
        memories = []
        # memories = deque([])
        model.eval()

        with jsonlines.open( os.path.join(self.args.output_dir, f"replacedInstruction{self.args.local_rank}.json"),"a") as writer:
            countReplaceInstruction = 0
            writer.write({"begin":f"-------------------epoch {episode}-------------------"})
            for step, inputs in tqdm(enumerate(epoch_iterator)):

            # inputs["max_length"] = model.config.max_length

            # TODO Temperature  inputs["temperature"] =self.args.temperature
            #

                inputs = self._prepare_inputs(inputs)

                generated_instructions, generated_rewards_last_instructions = model.generate_ppo(**inputs)

                instructions = self.tokenizer.batch_decode(generated_instructions, skip_special_tokens=True)
                instances = self.tokenizer.batch_decode(inputs["woinstruction_input_ids"], skip_special_tokens=True)

                ### before_logits
                generated_last_logits = model.get_actor_logits(input_ids=inputs["input_ids"],
                                                            attention_mask=inputs["attention_mask"],
                                                            labels=inputs["instruction_labels"],
                                                            )

                ### new logits

                generated_instruction_labels = self.tokenizer(instructions, padding="max_length", return_tensors="pt",truncation=True,
                                                            max_length=self.model.logits_shape)
                # print(instructions)
                # print(generated_instruction_labels["input_ids"].shape)
                label_mask = generated_instruction_labels["attention_mask"].bool()
                generated_instruction_labels = generated_instruction_labels["input_ids"].masked_fill(~label_mask, -100)
                generated_instruction_labels = self._prepare_inputs(generated_instruction_labels)
                generated_new_logits = model.get_actor_logits(input_ids=inputs["input_ids"].unsqueeze(dim=1).repeat_interleave(self.ppo_beam,dim=1).view(-1,inputs["input_ids"].shape[-1]),
                                                            attention_mask=inputs["attention_mask"].unsqueeze(dim=1).repeat_interleave(self.ppo_beam,dim=1).view(-1,inputs["attention_mask"].shape[-1]),
                                                            labels=generated_instruction_labels,
                                                            )
                # TODO Last time mask


                ## new instruction rewards
                generated_items = [instructions[i] +"\n\n"+ instances[int(i/self.ppo_beam)] for i in range(len(instructions))]

                generated_inputs = self.tokenizer(generated_items,padding=True ,truncation=True, return_tensors="pt",max_length=self.data_collator.max_source_length)
                generated_inputs["labels"] = inputs["labels"].unsqueeze(dim=1).repeat_interleave(self.ppo_beam,dim=1).view(-1,inputs["labels"].shape[-1])
                generated_inputs = self._prepare_inputs(generated_inputs)
                generated_rewards_newest_instructions = model.get_rewards(**generated_inputs)

                max_rewards_newest_indices = generated_rewards_newest_instructions.view(-1,self.ppo_beam).max(dim=-1)[1]

                for i in range(len(generated_rewards_last_instructions)):

                    item = inputs["instructed_input_ids"][i]
                    instance = inputs["input_ids"][i]
                    instruction = inputs["instruction_labels"][i]
                    reward = generated_rewards_last_instructions.unsqueeze(dim=-1)[i]
                    logit = generated_last_logits[i]
                    mask = (inputs["instruction_labels"][i]!=-100)

                    # set True for optimize instruction generation
                    if True :
                    # if generated_rewards_newest_instructions[i].item() > generated_rewards_last_instructions[i].item() and generated_rewards_newest_instructions[i].item()>0.1:
                        newest_index = self.ppo_beam * i + max_rewards_newest_indices[i]
                        item = generated_inputs["input_ids"][newest_index]
                        instruction = generated_instruction_labels[newest_index]
                        reward = generated_rewards_newest_instructions.unsqueeze(dim=-1)[newest_index]
                        logit = generated_new_logits[newest_index]
                        mask = (generated_instruction_labels[newest_index]!=-100)
                        
                        writer.write(
                            {"Original":self.tokenizer.decode(inputs["instruction_labels"][i].masked_fill(inputs["instruction_labels"][i]==-100, self.tokenizer.pad_token_id),skip_special_tokens=True),
                            "OriginalScore":generated_rewards_last_instructions[i].tolist(),
                            "Generated":instructions[newest_index],
                            "GeneratedScore":generated_rewards_newest_instructions[newest_index].tolist()
                            }
                        )

                    # memories["instance"].append(instance.detach().cpu().tolist())
                    # memories["instruction"].append(instruction.detach().cpu().tolist())
                    # memories["item"].append(item.detach().cpu().tolist())
                    # memories["labels"].append(inputs["labels"][i].detach().cpu().tolist())
                    # memories["reward"].append(reward.detach().cpu().tolist())
                    # memories["logits"].append(logit.detach().cpu().tolist())
                    # memories["logits_mask"].append(mask.detach().cpu().tolist())

                    # memories.append(
                    #     Memory(
                    #         *map(
                    #             lambda x: x.detach().cpu(),
                    #             (
                    #                 instance,
                    #                 instruction,
                    #                 item,
                    #                 inputs["labels"][i],  # diversity
                    #                 reward,
                    #                 logit,
                    #                 mask,
                    #             ),
                    #         )
                    #     )
                    # )
                    # "labels":self.tokenizer.batch_decode(inputs["labels"][i].masked_fill(inputs["labels"]==-100, self.tokenizer.pad_token_id).detach().cpu(),skip_special_tokens=True),
                    assert len(inputs["labels"][i].shape)==1,f"labels assert {inputs['labels'][i].shape}"
                    assert len(reward.shape)==2,f"reward assert{reward.shape}{generated_rewards_last_instructions}"
                    assert len(logit.shape)==1,f"logit assert{logit.shape}"
                    assert len(mask.shape)==1,f"logits_mask assert{mask.shape}"
                    assert len(instruction.shape)==1,f"instruction assert{instruction.shape}"
                    
                    memory = {
                        "instance":[self.tokenizer.decode(instance.detach().cpu(),skip_special_tokens=True)],
                        "instruction":instruction.squeeze(dim=0).detach().cpu(),
                        "item":[self.tokenizer.decode(item.detach().cpu(),skip_special_tokens=True)],
                        "labels": inputs["labels"][i].detach().cpu(),
                        "reward":reward.detach().cpu().float(),
                        "logits":logit.detach().cpu().float(),
                        "logits_mask":mask.detach().cpu(),
                    }

                    if torch.distributed.is_available() and torch.distributed.is_initialized(): 
                        memory_list = [None for i in range(torch.distributed.get_world_size())]
                        torch.distributed.all_gather_object(memory_list, memory)
                        memories.extend(
                            memory_list 
                        )
                    else:
                        memories.append(memory)

                    # print(memory_list)
                if step >= self.args.max_rl_sample:
                    break
        return memories

    def evaluation_loop(
            self,
            dataloader: DataLoader,
            description: str,
            prediction_loss_only: Optional[bool] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train init deepspeed here
        if args.deepspeed and not self.deepspeed:
            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(
                self, num_training_steps=0, resume_from_checkpoint=None
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine

        model = self._wrap_model(self.model, training=False)

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = dataloader.batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader.dataset):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = dataloader.dataset

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [args.device]).per_device_loader(args.device)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

            if is_torch_tpu_available():
                xm.mark_step()

            # Update containers on host
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if labels is not None:
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, labels_host = None, None, None

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and hasattr(eval_dataset, "num_examples"):
            num_samples = eval_dataset.num_examples
        else:
            num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            metrics = self.compute_metrics(dataset=eval_dataset, preds=all_preds, save_prefix=metric_key_prefix)
        else:
            metrics = {}

        metrics["global_step"] = self.state.global_step

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        # gen_kwargs = {
        #     "max_length": self._max_length if self._max_length is not None else self.model.config.max_length,
        #     "num_beams": self._num_beams if self._num_beams is not None else self.model.config.num_beams,
        #     "synced_gpus": True if is_deepspeed_zero3_enabled() else False,
        # }

        gen_kwargs = self._gen_kwargs.copy()
        if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
            gen_kwargs["max_length"] = self.model.config.max_length
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"] if gen_kwargs.get("num_beams") is not None else self.model.config.num_beams
        )
        default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
        gen_kwargs["synced_gpus"] = (
            gen_kwargs["synced_gpus"] if gen_kwargs.get("synced_gpus") is not None else default_synced_gpus
        )
        # print(gen_kwargs["max_length"])

        if not self.args.bi_generate:
            if "instructed_input_ids" in inputs:
                gen_kwargs["input_ids"] = inputs.get("instructed_input_ids", None)
            if "instructed_attention_mask" in inputs:
                gen_kwargs["attention_mask"] = inputs.get("instructed_attention_mask", None)
            generated_tokens = self.model.generate(
                **gen_kwargs,
            )
        else:
            instruction_gen_kwargs = self._gen_kwargs.copy()

            if instruction_gen_kwargs.get("max_length") is None and instruction_gen_kwargs.get("max_new_tokens") is None:
                instruction_gen_kwargs["max_length"] = self.model.config.max_length
            instruction_gen_kwargs["num_beams"] = (
                instruction_gen_kwargs["num_beams"] if instruction_gen_kwargs.get("num_beams") is not None else self.model.config.num_beams
            )
            default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
            instruction_gen_kwargs["synced_gpus"] = (
                instruction_gen_kwargs["synced_gpus"] if instruction_gen_kwargs.get("synced_gpus") is not None else default_synced_gpus
            )

            if "input_ids" in inputs:
                instruction_gen_kwargs["input_ids"] = inputs.get("input_ids", None)
            if "instructed_attention_mask" in inputs:
                instruction_gen_kwargs["attention_mask"] = inputs.get("attention_mask", None)

            # prepare generation inputs
            # some encoder-decoder models can have varying encoder's and thus
            # varying model input names
            # if hasattr(self.model, "encoder") and self.model.encoder.main_input_name != self.model.main_input_name:
            #     generation_inputs = inputs[self.model.encoder.main_input_name]
            #     gen_kwargs[self.model.encoder.main_input_name] = inputs[self.model.encoder.main_input_name]
            # else:
            #     generation_inputs = inputs[self.model.main_input_name]
            #     gen_kwargs[self.model.main_input_name] = inputs[self.model.encoder.main_input_name]
            generated_instructions = self.model.generate_instruction(
                    **instruction_gen_kwargs
                 )

            instructions = self.tokenizer.batch_decode(generated_instructions, skip_special_tokens=True)
            instances = self.tokenizer.batch_decode(inputs["woinstruction_input_ids"], skip_special_tokens=True)

            generated_items = [instructions[i] + "\n\n" + instances[i] for i in range(len(instructions))]
            generated_inputs = self.tokenizer(generated_items, padding=True, truncation=True, return_tensors="pt")
            generated_inputs = self._prepare_inputs(generated_inputs)
            gen_kwargs["input_ids"] = generated_inputs["input_ids"]
            gen_kwargs["attention_mask"] = generated_inputs["attention_mask"]
            generated_tokens = self.model.generate(
                **gen_kwargs,
            )

        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        with torch.no_grad():
            if has_labels:
                with self.autocast_smart_context_manager():
                    outputs = model(**inputs)
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_kwargs["max_length"]:
                labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])
        else:
            labels = None

        return (loss, generated_tokens, labels)
