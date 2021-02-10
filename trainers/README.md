# Trainers
Containing the core execution loop of your program, trainers are where the models learn.

## Base Class
All trainers inherit from `BaseTrainer` in `\trainers\base.py`.

### Standard Methods
- `.train()` is the only method your trainer needs to have, and should complete an entire round of training.

## Relation to Logging
Trainers are instantiated with access to a logger. It's the trainer's responsibility to add values to the logger, which
will then perform any advanced postprocessing, plotting, and saving to disk the next time `.post()` is called. The
trainer is also the one calling `.post()` via the logger, at every point in training you want to save to disk. It's
advised this be done intermittently, so that not too much information is being held in memory.
