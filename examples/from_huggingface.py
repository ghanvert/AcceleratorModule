from accmt import Trainer, HyperParameters, DataCollatorForSeq2Seq
from dummy_dataset import DummyTranslationDataset
from transformers import NllbTokenizer

hf_model = "facebook/nllb-200-distilled-600M"

tokenizer = NllbTokenizer.from_pretrained(hf_model)
train_dataset = DummyTranslationDataset(tokenizer)

trainer = Trainer(
    hps_config=HyperParameters(),
    model_path="dummy_model",
    collate_fn=DataCollatorForSeq2Seq(tokenizer)
    # adds efficient padding to the longest sequence in the batch.
)
trainer.fit((hf_model, "AutoModelForSeq2SeqLM"), train_dataset)
