import numpy as np

from abc import ABC
from accelerate.utils import LoggerType, ProjectConfiguration
from config import read, save_status, read_status


class AcceleratorModule(ABC):
    def forward(self, x):
        pass
    
    def training_step(self, batch):
        pass
    
    def validation_step(self, batch):
        pass
    
    def __init_subclass__(cls, **kwargs):
        if cls.training_step == AcceleratorModule.training_step and cls.validation_step == AcceleratorModule.validation_step:
            raise TypeError(
                "Subclasses of 'Trainer' must override 'training_step' and/or 'validation_step' methods."
            )
        super().__init_subclass__(**kwargs)

    def __call__(self, *args):
        return self.forward(*args)

    def __repr__(self):
        return self.model
    
    def __str__(self):
        return self.model.__repr__()
    
    def __len__(self):
        return sum(p.numel() for p in self.model.parameters())

class Trainer:
    def __init__(self,
                accelerator,
                hps_file_config: str = None,
                checkpoint = "checkpoint1",
                resume = False,
                model_path: str = None,
                model_saving = "best_valid_loss",
                enable_checkpointing = True,
                checkpoint_every=1,
                logging_dir = "logs",
                log_with = LoggerType.TENSORBOARD
    ):
        self.accelerator = accelerator
        self.hps_config = hps_file_config
        self.checkpoint = checkpoint
        self.resume = resume
        self.model_path = model_path
        self.model_saving = model_saving.lower()
        self.enable_checkpointing = enable_checkpointing
        self.checkpoint_every = checkpoint_every
        self.logging_dir = logging_dir

        self.accelerator.project_configuration = ProjectConfiguration(project_dir=".", logging_dir=logging_dir, total_limit=1)
        self.accelerator.log_with = [log_with]

    def fit(self,
            module: AcceleratorModule,
            train_dataset,
            val_dataset = None
    ):
        import os
        import torch

        from tqdm.auto import tqdm
        from torch.utils.data import DataLoader

        model = getattr(module, "model", None)
        
        if not model:
            raise AttributeError("'self.model' needs to be declared in the AcceleratorModule class.")
        
        cfg = read(self.hps_config)
        hps = cfg["hps"]
        optim = hps["optim"]
        schlr = hps["scheduler"]

        if not self.model_path:
            self.model_path = cfg["version"]

        if self.model_saving:
            os.makedirs(self.model_path, exist_ok=True)

        train_dataloader = DataLoader(train_dataset, batch_size=hps["batch_size"], shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=hps["batch_size"], shuffle=True)
        
        optimizer = getattr(torch.optim, optim["type"])(model.parameters(), lr=float(optim["lr"]), weight_decay=float(optim["weight_decay"]))
        scheduler = None
        if "type" in schlr:
            scheduler = getattr(torch.optim.lr_scheduler, schlr["type"])(optimizer, max_lr=float(schlr["max_lr"]), steps_per_epoch=len(train_dataloader), epochs=hps["epochs"])

        model, train_dataloader, val_dataloader, optimizer, scheduler = self.accelerator.prepare(
            model, train_dataloader, val_dataloader, optimizer, scheduler
        )
        self.accelerator.init_trackers(cfg["version"])

        best_valid_loss = float("inf")
        best_train_loss = float("inf")
        status_epoch = 0

        if self.resume:
            if os.path.exists(self.checkpoint):
                self.accelerator.load_state(self.checkpoint)
                status = read_status(f"{self.checkpoint}/status.json")
                best_valid_loss = status["best_valid_loss"]
                best_train_loss = status["best_train_loss"]
                status_epoch = status["epoch"]
            else:
                print(f"{self.checkpoint} does not exist. Starting process from zero...")

        epochs = hps["epochs"]
        global_step = 0
        eval_step = len(train_dataloader) // len(val_dataloader)
        for epoch in range(status_epoch, epochs):
            eval_global_step = global_step
            model.train()
            train_losses = []
            for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch}/{epochs}", unit="batch"):
                loss = module.training_step(batch)

                train_losses.append(loss.item())
                if step % cfg["log_every"] == 0:
                    self.accelerator.log({"loss": {"train": loss.item()}}, step=global_step)

                self.accelerator.backward(loss)
                optimizer.step()
                if scheduler:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1
            
            if all([val_dataset, getattr(module, "validation_step", False)]):
                model.eval()
                eval_losses = []
                with torch.no_grad():
                    for step, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc=f"Epoch {epoch}/{epochs}", unit="batch"):
                        loss = module.validation_step(batch)

                        eval_losses.append(loss.item())
                        if step % cfg["log_every"] == 0:
                            self.accelerator.log({"loss": {"valid": loss.item()}}, step=eval_global_step)

                        eval_global_step += eval_step

                self._save_model_on_criteria(model, eval_losses, train_losses, best_valid_loss, best_train_loss)
            else:
                if self.model_saving:
                    avg_train_loss = np.mean(train_losses)
                    if avg_train_loss < best_train_loss:
                        self._save_model(model, best_valid_loss, best_train_loss)

            if self.enable_checkpointing and epoch % self.checkpoint_every == 0:
                self.accelerator.save_state(self.checkpoint)
                save_status({
                    "best_valid_loss": best_valid_loss,
                    "best_train_loss": best_train_loss,
                    "epoch": epoch
                }, to=f"{self.checkpoint}/status.json")

        self.accelerator.end_training()

    
    def _save_model(self, model, best_valid_loss, best_train_loss):
        state_dict = self.accelerator.get_state_dict(model)
        unwrapped_model = self.accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            self.model_path,
            is_main_process=self.accelerator.is_main_process,
            state_dict=state_dict,
            max_shard_size="10GB",
            save_function=self.accelerator.save,
            safe_serialization=False # if True, some tensors will not be saved
        )

        save_status({
            "best_valid_loss": best_valid_loss,
            "best_train_loss": best_train_loss,
        }, to=f"{self.model_path}/status.json")

    
    def _save_model_on_criteria(self, model, eval_losses, train_losses, best_valid_loss, best_train_loss):
        if self.model_saving is None:
            return

        saving_criteria = {
            "best_valid_loss": (np.mean(eval_losses) < best_valid_loss),
            "best_train_loss": (np.mean(train_losses) < best_train_loss),
            "always": True
        }

        if self.model_saving in saving_criteria:
            if saving_criteria[self.model_saving]:
                self._save_model(model, best_valid_loss, best_train_loss)
        else:
            raise ValueError("Invalid type of model saving. Value must be: "
                              "'best_valid_train_loss', "
                              "'best_train_loss', or "
                              "'always'.")