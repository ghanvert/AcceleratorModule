from accelerate import Accelerator
from argparse import ArgumentParser
from dataset import RapSpaDataset
from trainer import AcceleratorModule, Trainer
from transformers import NllbTokenizer, AutoModelForSeq2SeqLM


parser = ArgumentParser(description="Train or finetune a model.")
parser.add_argument("--hf-model", type=str, help="🤗 HuggingFace model.")
parser.add_argument("--config", type=str, help="YAML config file to train this model.")
parser.add_argument("--checkpoint", type=str, help="Checkpoint path.")
args = parser.parse_args()
hf_model = args.hf_model
config = args.config
checkpoint = args.checkpoint

accelerator = Accelerator()

tokenizer = NllbTokenizer.from_pretrained(hf_model)

class Module(AcceleratorModule):
    def __init__(self):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(hf_model)
        self.pad_token_id = tokenizer.pad_token_id

    def training_step(self, batch):
        x, y = batch
        y['input_ids'][y['input_ids'] == self.pad_token_id] = -100
        return self.model(**x, labels=y.input_ids).loss
    
    def validation_step(self, batch):
        x, y = batch
        y['input_ids'][y['input_ids'] == self.pad_token_id] = -100
        return self.model(**x, labels=y.input_ids).loss
    

train_dataset = RapSpaDataset('dataset/train.jsonl', "spa_Latn", "mri_Latn", tokenizer=tokenizer, mix=False)
val_dataset = RapSpaDataset('dataset/val.jsonl', "spa_Latn", "mri_Latn", tokenizer=tokenizer, mix=False)

module = Module()
trainer = Trainer(
    accelerator,
    hps_file_config=config,
    checkpoint=checkpoint
)

trainer.fit(module, train_dataset, val_dataset)