from transformers import TrainingArguments, Trainer, logging, AutoModelForSequenceClassification
import numpy as np
from .utils.GPUUtilization import print_gpu_utilization,print_summary
from .utils.pandoraUtils import PandoraUtils
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from torch import cuda
from datetime import datetime
import time

default_args = {
        "evaluation_strategy": "epoch",
        #"evaluation_strategy": "steps",
        #"eval_steps": 0.001,   # 10 per epoch
        #"logging_steps": 12.5,
        "num_train_epochs": 5,
        "log_level": "error",
        "report_to": "none",
        "output_dir": "model_checkpoints/" + datetime.now().strftime("%Y%m%d_%H%M%S"),
        "save_strategy": "epoch",
        "fp16": True,
        "per_device_train_batch_size": 16,
        "gradient_accumulation_steps": 4,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_accuracy",
        "optim": "adamw_torch",
        "learning_rate": 3e-5,
        "lr_scheduler_type": "linear",
        "weight_decay": 0.1,
        "adam_epsilon": 1e-8,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
    }

device = 'cuda' if cuda.is_available() else 'cpu'


def run_one_trait_pandora(trait_name):
    logging.set_verbosity_error()

    # Get dataset
    desired_size = 92800 # 80% of 116000
    tokenized_dataset = PandoraUtils.get_tokenized_dataset_one_trait(trait_name,desired_size,0.2)

    # Load model, set 2 labels, send to GPU
    #model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2).to(device)
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-cased", num_labels=2).to(device)

    # Set training arguments
    training_args = TrainingArguments(seed=17, **default_args)

    # Set metrics for evaluation
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        accuracy = accuracy_score(y_true=labels, y_pred=predictions)
        recall = recall_score(y_true=labels, y_pred=predictions)
        precision = precision_score(y_true=labels, y_pred=predictions)
        f1 = f1_score(y_true=labels, y_pred=predictions)

        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, 
                "time_passed": str(time.perf_counter() - trainint_start) + "s"}

    # Freeze 0-4 layers, we will train 5 and new classification layer
    freeze_layers = ["distilbert.transformer.layer.0", "distilbert.transformer.layer.1", "distilbert.transformer.layer.2", "distilbert.transformer.layer.3", "distilbert.transformer.layer.4"]
    for name, param in model.named_parameters():
        # Check if name starts with one of the layers we want to freeze
        if any(name.startswith(layer) for layer in freeze_layers):
            param.requires_grad = False

    # Correct freeze check
    #for name, param in model.named_parameters():
    #    print(name, param.requires_grad)

    # Define trainer (for adding learning rate to logs)
    class MyTrainer(Trainer):
        def log(self, logs: "dict[str, float]") -> None:
            logs["learning_rate"] = self._get_learning_rate()
            super().log(logs)
    
    trainer = MyTrainer(model=model, args=training_args, compute_metrics=compute_metrics, train_dataset=tokenized_dataset["train"],
                       eval_dataset=tokenized_dataset["test"])
    
    # Train
    print("\n\nStarting training - " + datetime.now().strftime("%Y%m%d_%H%M%S") + "\n\n")
    trainint_start = time.perf_counter()
    result = trainer.train()

    # Save model
    trainer.save_model("saved_models/" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_datasetsize" + str(desired_size) + 
                       "_epochs" + str(training_args.num_train_epochs) + "_" + trait_name + "_pandora")
    
    print_summary(result)


def run_kcross_one_trait_pandora(trait_name, k):
    logging.set_verbosity_error()

    # Set dataset size
    desired_size = 92800 # 80% of 116000

    # Set metrics for evaluation
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        accuracy = accuracy_score(y_true=labels, y_pred=predictions)
        recall = recall_score(y_true=labels, y_pred=predictions)
        precision = precision_score(y_true=labels, y_pred=predictions)
        f1 = f1_score(y_true=labels, y_pred=predictions)

        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, 
                "time_passed": str(time.perf_counter() - training_start) + "s"}
    
    # Define trainer (for adding learning rate to logs)
    class MyTrainer(Trainer):
        def log(self, logs: "dict[str, float]") -> None:
            logs["learning_rate"] = self._get_learning_rate()
            super().log(logs)

    start_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")

    # For each k, load dataset, load model, train
    for curr_k in range(k):
        tokenized_dataset = PandoraUtils.get_one_kcross_tokenized_dataset_one_trait(trait_name,desired_size, k, curr_k)

        # Load model, set 2 labels, send to GPU
        #model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2).to(device)
        model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-cased", num_labels=2).to(device)

        # Set training arguments
        default_args["output_dir"] = "model_checkpoints/" + start_datetime + "_k" + str(curr_k)
        training_args = TrainingArguments(**default_args)

        # Freeze 0-4 layers, we will train 5 and new classification layer
        freeze_layers = ["distilbert.transformer.layer.0", "distilbert.transformer.layer.1", "distilbert.transformer.layer.2", "distilbert.transformer.layer.3", "distilbert.transformer.layer.4"]
        for name, param in model.named_parameters():
            # Check if name starts with one of the layers we want to freeze
            if any(name.startswith(layer) for layer in freeze_layers):
                param.requires_grad = False

        trainer = MyTrainer(model=model, args=training_args, compute_metrics=compute_metrics, train_dataset=tokenized_dataset["train"],
                        eval_dataset=tokenized_dataset["test"])
        
        # Train
        print("\n\nStarting training {0} - ".format(curr_k) + datetime.now().strftime("%Y%m%d_%H%M%S") + "\n\n")
        training_start = time.perf_counter()
        result = trainer.train()

        # Save model
        trainer.save_model("saved_models/" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_datasetsize" + str(desired_size) + 
                        "_epochs" + str(training_args.num_train_epochs) + "_" + trait_name + "_pandora_currk" + str(curr_k) )
        
        print_summary(result)

