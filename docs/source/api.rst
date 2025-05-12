API Reference
============

Trainer
-----------

.. autoclass:: accmt.Trainer
   :members: __init__, fit, register_model_saving, log_artifact, log_artifacts
   :undoc-members: False
   :noindex:


AcceleratorModule
------------------

.. autoclass:: accmt.AcceleratorModule
   :members: training_step, validation_step, get_optimizer, get_train_dataloader, get_validation_dataloader, log, __call__, forward, __len__, pad, freeze, unfreeze
   :undoc-members: False
   :noindex:


ExtendedAcceleratorModule
-------------------------

.. autoclass:: accmt.ExtendedAcceleratorModule
   :members: backward, step_optimizer, step_scheduler, step, zero_grad
   :undoc-members: False
   :noindex:


States
------

.. autoclass:: accmt.states.TrainingState
   :members: additional_metrics, best_train_loss, epoch, evaluations_done, finished, global_step, is_end_of_epoch, is_last_epoch, is_last_training_batch, is_last_validation_batch, num_checkpoints_made, patience_left, train_step, val_step
   :undoc-members: False
   :noindex:


Callbacks
-------

.. autoclass:: accmt.callbacks.Callback
   :members:
   :undoc-members: False
   :noindex:


Metrics
-------

.. autoclass:: accmt.metrics.Metric
   :members: __init__, compute
   :undoc-members: False
   :noindex:

.. autoclass:: accmt.metrics.MetricParallel
   :members: __init__, compute
   :undoc-members: False
   :noindex:


DataCollators
------------

.. autoclass:: accmt.collate_fns.DataCollatorForSeq2Seq
   :members: __init__
   :undoc-members: False
   :noindex:

.. autoclass:: accmt.collate_fns.DataCollatorForLanguageModeling
   :members: __init__
   :undoc-members: False
   :noindex:

.. autoclass:: accmt.collate_fns.DataCollatorForLongestSequence
   :members: __init__
   :undoc-members: False
   :noindex:


Monitor
-------

.. autoclass:: accmt.monitor.Monitor
   :members: __init__
   :undoc-members: False
   :noindex:


HyperParameters
--------------

.. autoclass:: accmt.hyperparameters.HyperParameters
   :members: __init__
   :undoc-members: False
   :noindex:


Optimizers
----------

.. autoclass:: accmt.hyperparameters.Optimizer
   :members:
   :undoc-members: False
   :noindex:


Schedulers
----------

.. autoclass:: accmt.hyperparameters.Scheduler
   :members:
   :undoc-members: False
   :noindex:


HyperParameterSearch
---------------------

.. autoclass:: accmt.hp_search.HyperParameterSearch
   :members: __init__, set_parameters, optimize
   :undoc-members: False
   :noindex: