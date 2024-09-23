from transformers.trainer import Trainer
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, RandomSampler, SequentialSampler
from transformers import (DataCollator, PreTrainedModel, Trainer, PreTrainedTokenizerBase, EvalPrediction, TrainingArguments, TrainerCallback)


class BothEvalTrainer(Trainer):
    
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset, "datasets.Dataset"]] = None,
        eval_dataset1: Optional[Union[Dataset, Dict[str, Dataset], "datasets.Dataset"]] = None,
        eval_dataset2: Optional[Union[Dataset, Dict[str, Dataset], "datasets.Dataset"]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__(model=model,
                         args=args,
                         data_collator=data_collator,
                         train_dataset=train_dataset,
                         eval_dataset=eval_dataset1,
                         tokenizer=tokenizer,
                         model_init=model_init,
                         compute_metrics=compute_metrics,
                         callbacks=callbacks,
                         optimizers=optimizers,
                         preprocess_logits_for_metrics=preprocess_logits_for_metrics)
        self.eval_dataset1 = eval_dataset1
        self.eval_dataset2 = eval_dataset2
    
    
    def _evaluate(self, trial, ignore_keys_for_eval, skip_scheduler=False):
        metrics1 = self.evaluate(eval_dataset=self.eval_dataset1,ignore_keys=ignore_keys_for_eval)
        self._report_to_hp_search(trial, self.state.global_step, metrics1)
        metrics2 = self.evaluate(eval_dataset=self.eval_dataset2,ignore_keys=ignore_keys_for_eval)
        self._report_to_hp_search(trial, self.state.global_step, metrics2)

        # Run delayed LR scheduler now that metrics are populated
        if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) and not skip_scheduler:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            try:
                self.lr_scheduler.step(metrics1[metric_to_check])
            except KeyError as exc:
                raise KeyError(
                    f"The `metric_for_best_model` training argument is set to '{metric_to_check}', which is not found in the evaluation metrics. "
                    f"The available evaluation metrics are: {list(metrics1.keys())}. Consider changing the `metric_for_best_model` via the TrainingArguments."
                ) from exc
        return metrics1