# Custom Trainer with proper metric calculation
from transformers import Trainer
import torch
import numpy as np

class MyTrainer(Trainer):
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        # Calling the original evaluate method
        eval_results = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # Perform evaluation manually if necessary (e.g., to retrieve logits and labels)
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        all_preds, all_labels = [], []

        for batch in eval_dataloader:
            with torch.no_grad():
                inputs = {key: value.to(self.args.device) for key, value in batch.items() if key != "labels"}
                outputs = self.model(**inputs)
                logits = outputs.logits
                labels = batch["labels"].to(self.args.device)
                preds = torch.argmax(logits, dim=-1)

                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        # Compute metrics
        if self.compute_metrics is not None:
            metrics = self.compute_metrics((all_preds, all_labels))
        
        eval_results.update(metrics)
        
        return eval_results