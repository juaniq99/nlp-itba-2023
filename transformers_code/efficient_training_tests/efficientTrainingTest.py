from pynvml import *
from transformers import TrainingArguments, Trainer, logging, AutoModelForSequenceClassification
import numpy as np
from datasets import Dataset


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()



seq_len, dataset_size = 512, 512
dummy_data = {
    "input_ids": np.random.randint(100, 30000, (dataset_size, seq_len)),
    "labels": np.random.randint(0, 1, (dataset_size)),
}
ds = Dataset.from_dict(dummy_data)
ds.set_format("pt")


default_args = {
    "output_dir": "tmp",
    "evaluation_strategy": "steps",
    "num_train_epochs": 1,
    "log_level": "error",
    "report_to": "none",
}

logging.set_verbosity_error()

print_gpu_utilization()

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased").to("cuda")
print_gpu_utilization()

#training_args = TrainingArguments(per_device_train_batch_size=4, **default_args) # 20.27s, 4112MB
#training_args = TrainingArguments(per_device_train_batch_size=1, gradient_accumulation_steps=4, **default_args) # 19.47s, 3226MB
#training_args = TrainingArguments(per_device_train_batch_size=4, fp16=True, **default_args) # 10.21s 3942MB
#training_args = TrainingArguments(per_device_train_batch_size=4, gradient_accumulation_steps=4, fp16=True, **default_args) # 8.8s 4047MB
training_args = TrainingArguments(per_device_train_batch_size=8, gradient_accumulation_steps=8, fp16=True, **default_args) # 7.19 4851MB

trainer = Trainer(model=model, args=training_args, train_dataset=ds)
result = trainer.train()
print_summary(result)