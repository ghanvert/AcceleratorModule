Usage
=====

Basic Usage
----------

.. code-block:: python

    from accmt import AcceleratorModule, Trainer, HyperParameters

    class ExampleModule(AcceleratorModule):
        def __init__(self):
            self.model = ...
            # self.model is required.

        def training_step(self, batch):
            x, y = batch
            # ...
            return train_loss

        def validation_step(self, key, batch):
            x, y = batch
            # ...
            return {
                "loss": val_loss,
                # any other metric...
            }

    if __name__ == "__main__":
        module = ExampleModule()

        trainer = Trainer(
            hps_config=HyperParameters(epochs=2),
            model_path="model_folder",
        )

        train_dataset = ...
        val_dataset = ...

        trainer.fit(module, train_dataset, val_dataset)

To run training on multiple GPUs, you can use the following command:

.. code-block:: bash

    accmt launch train.py


Advanced Usage
-------------

For more advanced usage, please refer to the API documentation. 