# Trainer for HuggingFace

ACCMT Trainer supports training for standard HuggingFace models. You do not need to build a module for every model, since every model has a standard way to be constructed (we construct this in the background). The only thing you need to take care of is your Dataset logic, since the inner **forward** function of a model might take some different arguments.

## Using AcceleratorModule to automatically construct the module

```python
from accmt import AcceleratorModule

module = AcceleratorModule.from_hf("PATH_TO_HF_MODEL", "TRANSFORMERS_CLASS_TYPE", ...)
# "..." might contain arguments for "from_pretrained" function from transformers library.

# example:
# module = AcceleratorModule.from_hf("facebook/nllb-200-3.3B", "AutoModelForSeq2Seq", torch_dtype=torch.bfloat16)
```

## Using fit function to automatically train the module

You can also construct the module inside of the **fit** function from the Trainer class.

```python
from accmt import Trainer

trainer = Trainer(...)
trainer.fit(("PATH_TO_HF_MODEL", "TRANSFORMERS_CLASS_TYPE"), train_dataset, val_dataset, ...)
# "..." might contain arguments for "from_pretrained" function from transformers library.
```
