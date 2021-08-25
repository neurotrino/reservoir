"""Sample of the two idiomatic ways to train.

tensorflow.org/tutorials/customization/custom_training_walkthrough
"""

from tensorflow.keras.utils import Progbar

import logging
import numpy as np
import os
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.profiler.experimental as profiler

# local
from trainers.base import BaseTrainer

from models.common import fano_factor

class Trainer(BaseTrainer):
    """TODO: docs  | note how `optimizer` isn't in the parent"""

    def __init__(self, cfg, model, data, logger):
        super().__init__(cfg, model, data, logger)

        train_cfg = cfg['train']

        try:
            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate=train_cfg.learning_rate
            )
        except Exception as e:
            logging.warning(f"learning rate not set: {e}")


    #┬───────────────────────────────────────────────────────────────────────╮
    #┤ Training Loop (step level)                                            │
    #┴───────────────────────────────────────────────────────────────────────╯

    @tf.function
    def loss(self, x, y):
        """Calculate the loss on data x labeled y."""
        loss_object = tf.keras.losses.MeanSquaredError()

        # [*] If you want per-trial logging values when using batches,
        # you'll have to iterate over the x and y values, instead of
        # just calling model with them once. Another option is to
        # implement `.call()` in you model such that, when performing
        # these operations, it returns a list of values, which would be
        # trial values, plus the end values; or `.call()` could take a
        # logger as an optional argument.

        voltage, spikes, prediction = self.model(x) # tripartite output
        task_loss = loss_object(y_true=y, y_pred=prediction)

        unitwise_rates = tf.reduce_mean(spikes, axis=(0, 1))
        rate_loss = tf.reduce_sum(tf.square(unitwise_rates - self.cfg['train'].target_rate)) * self.cfg['train'].rate_cost
        #interm_loss_val_0 = tf.math.add(task_loss,rate_loss)

        #synchrony = fano_factor(self, self.cfg['data'].seq_len, spikes)
        #synch_loss = tf.reduce_sum(tf.square(synchrony - self.cfg['train'].target_synch)) * self.cfg['train'].synch_cost
        #interm_loss_val_1 = tf.math.add(interm_loss_val_0,synch_loss)
        #total_loss_val = tf.math.add(interm_loss_val_0,synch_loss)
        #total_loss_val = tf.math.add(task_loss, synch_loss)
        total_loss_val = tf.math.add(task_loss, rate_loss)

        #w_rec = np.array(self.model.cell.recurrent_weights)

        #total_ct = np.size(w_rec)
        #zero_ct = w_rec[w_rec==0].shape[0]
        #dens = (total_ct - zero_ct)/float(total_ct)

        #conn_loss = tf.reduce_sum(tf.square(dens - self.cfg['train'].target_conn)) * self.cfg['train'].conn_cost
        #total_loss_val = tf.math.add(interm_loss_val_1,conn_loss)

        # [*] Because this is a tf.function, we can't collapse tensors
        # to numpy arrays for logging, so we need to return the tensors
        # then call `.numpy()` in `.train_step()`.

        # training=training is needed only if there are layers with
        # different behavior during training versus inference
        # (e.g. Dropout).

        return voltage, spikes, prediction, total_loss_val


    @tf.function
    def grad(self, inputs, targets):
        """Gradient calculation(s)"""
        with tf.GradientTape() as tape:
            voltage, spikes, prediction, loss_val = self.loss(inputs, targets)

        # Calculate the gradient of the loss with respect to each
        # layer's trainable variables. In this example, calculates the
        # gradients for (in order):
        # > `rnn/ex_in_lif/input_weights:0`
        # > `rnn/ex_in_lif/recurrent_weights:0`
        # > `dense/kernel:0`
        # > `dense/bias:0`
        grads = tape.gradient(loss_val, self.model.trainable_variables)
        return voltage, spikes, prediction, loss_val, grads


    def train_step(self, batch_x, batch_y, batch_idx=None):
        """Train on the next batch."""

        # [?] Are we saying that each batch steps with dt?

        #┬───────────────────────────────────────────────────────────────────╮
        #┤ Pre-Step Logging                                                  │
        #┴───────────────────────────────────────────────────────────────────╯

        self.logger.on_step_begin()

        # Input/reference variables
        self.logger.log(
            data_label='inputs',
            data=batch_x.numpy(),
            meta={
                'stride': 'step',
                'shape_key': (
                    'num_batches * post_every',
                    'batch_size',
                    'seq_len',
                    'n_input'
                ),
                'description':
                    'inputs'
            }
        )
        self.logger.log(
            data_label='true_y',
            data=batch_y.numpy(),
            meta={
                'stride': 'step',
                'shape_key': (
                    'num_batches * post_every',
                    'batch_size',
                    'seq_len',
                    '1'
                ),
                'description':
                    'correct values'
            }
        )

        """
        # [!] empty first time (needs at least one forward pass)
        # [!] besides the first time, this is just postweights of the
        #     last batch, so we want to have a `static` save of the
        #     first time, but not this
        preweights = [x.numpy() for x in self.model.trainable_variables]
        """

        #┬───────────────────────────────────────────────────────────────────╮
        #┤ Gradient Calculation                                              │
        #┴───────────────────────────────────────────────────────────────────╯

        voltage, spikes, prediction, loss, grads = self.grad(batch_x, batch_y)

        #┬───────────────────────────────────────────────────────────────────╮
        #┤ Mid-Step Logging                                                  │
        #┴───────────────────────────────────────────────────────────────────╯

        # Output variables
        self.logger.log(
            data_label='voltage',
            data=voltage.numpy(),
            meta={
                'stride': 'step',
                'shape_key': (
                    'num_batches * post_every',
                    'batch_size',
                    'seq_len',
                    'n_recurrent'
                ),
                'description':
                    'voltages'
            }
        )
        self.logger.log(
            data_label='spikes',
            data=spikes.numpy(),
            meta={
                'stride': 'step',
                'shape_key':  (
                    'num_batches * post_every',
                    'batch_size',
                    'seq_len',
                    'n_recurrent'
                ),
                'description':
                    'spikes'
            }
        )
        self.logger.log(
            data_label='pred_y',
            data=prediction.numpy(),
            meta={
                'stride': 'step',
                'shape_key': (
                    'num_batches * post_every',
                    'batch_size',
                    'seq_len',
                    '1'
                ),
                'description':
                    'predictions'
            }
        )

        pre_zeros = tf.where(tf.equal(self.model.cell.recurrent_weights, 0))

        #┬───────────────────────────────────────────────────────────────────╮
        #┤ Gradient Application                                              │
        #┴───────────────────────────────────────────────────────────────────╯

        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables)
        )

        #┬───────────────────────────────────────────────────────────────────╮
        #┤ Sparsity Application                                              │
        #┴───────────────────────────────────────────────────────────────────╯

        # if freewiring is permitted (i.e. sparsity of any kind is NOT enforced),
        # this step is needed to explicitly ensure self-recurrent
        # connections remain zero otherwise (if sparsity IS enforced)
        # that is taken care of through the rec_sign application below
        if self.cfg['model'].cell.freewiring:
            # Make sure all self-connections remain 0
            self.model.cell.recurrent_weights.assign(tf.where(
                self.model.cell.disconnect_mask,
                tf.zeros_like(self.model.cell.recurrent_weights),
                self.model.cell.recurrent_weights
            ))

        # If the sign of a weight changed from the original or the
        # weight (previously 0) is no longer 0, make the weight 0.
        #
        # Reminder that rec_sign contains 0's for initial 0's when
        # rewiring = false whereas it contains +1's or -1's (for excit
        # or inhib) for initial 0's when rewiring = true
        self.model.cell.recurrent_weights.assign(tf.where(
            self.model.cell.rec_sign * self.model.cell.recurrent_weights > 0,
            self.model.cell.recurrent_weights,
            0
        ))

        if self.cfg['model'].cell.rewiring:
            #pre_zeros_ct = tf.cast(tf.size(pre_zeros)/2, tf.int32)
            post_zeros = tf.where(tf.equal(self.model.cell.recurrent_weights, 0))
            #post_zeros_ct = tf.where(tf.size(post_zeros)/2, tf.int32)
            #new_zeros_ct = tf.subtract(post_zeros_ct, pre_zeros_ct)
            new_zeros_ct = tf.subtract(tf.shape(post_zeros)[0],tf.shape(pre_zeros)[0])
            if new_zeros_ct > 0:
                for i in range(0,new_zeros_ct): # for all new zeros
                    # randomly select a position from post_zeros (total possible zeros)
                    new_pos_idx = np.random.randint(0, tf.shape(post_zeros)[0])
                    # draw a new weight
                    new_w = np.random.lognormal(self.model.cell.mu, self.model.cell.sigma)
                    if post_zeros[new_pos_idx][0] >= self.model.cell.n_excite:
                        # if inhib, make weight -10x
                        new_w = - new_w * 10
                    # reassign to self.recurrent_weights
                    #self.model.cell.recurrent_weights.assign(tf.where(
                        #self.model.cell.recurrent_weights == self.model.cell.recurrent_weights[tf.cast(post_zeros[new_pos_idx], tf.int32)],
                        #new_w,
                        #self.model.cell.recurrent_weights))
                    tf.tensor_scatter_nd_update(self.model.cell.recurrent_weights, [post_zeros[new_pos_idx]], [new_w])

        # In a similar way, one could use CMG to create sparse initial
        # input weights, then capture the signs so as to enforce
        # -1/+1/0's throughout training by doing
        # self.model.trainable_variables[0].assign() etc.


        #┬───────────────────────────────────────────────────────────────────╮
        #┤ Post-Step Logging                                                 │
        #┴───────────────────────────────────────────────────────────────────╯

        # [*] Log any step-wise variables for trainable variables
        #
        # These values are in different lists, but indexed the same,
        # i.e. `grad_data['names'][i]` will produce the name of the
        # layer with shape `grad_data['shapes'][i]`, and likewise for
        # values and gradients.
        for i in range(len(self.model.trainable_variables)):
            tvar = self.model.trainable_variables[i]

            # [?] should we store the names somewhere

            # Layer shape
            self.logger.log(  # [?] static value logged repeatedly
                data_label=tvar.name + '.shapes',
                data=tvar.shape,
                meta={
                    'stride': 'static',
                    'description':
                        'shape of layer ' + tvar.name
                }
            )

            # Calculated gradients
            self.logger.log(
                data_label=tvar.name + '.gradients',
                data=grads[i].numpy(),
                meta={
                    'stride': 'step',
                    'description':
                        'gradients calculated for ' + tvar.name
                }
            )

            """
            # Weights before applying gradients
            self.logger.log(
                data_label=tvar.name + '.preweights',
                data=preweights[i],
                meta={
                    'stride': 'step',
                    'description':
                        'layer weights for '
                        + tvar.name
                        + ' before applying the gradients'
                }
            )
            """

            # Weights after applying gradients
            try:
                self.logger.log(
                    data_label=f"tv{i}.postweights",
                    data=tvar.numpy(),
                    meta={
                        'stride': 'step',
                        'description':
                            'layer weights for '
                            + tvar.name
                            + ' after applying the gradients'
                    }
                )
            except:
                pass

        # [*] Stepwise logging for *all* layers; might have redundancy
        # [*] Can also grab layer weights here
        # [*] Use `if layer.name == "..."` to log with specific layers
        #
        # You can also perform this operation with the `layer.input`
        # attribute.
        #
        # Doing this for every layer at every step is really expensive,
        # because TensorFlow doesn't actually want you to collapse
        # tensor states like this (but does not currently provide a
        # cost-effective API to access these values any another way)
        #
        # If you can, it's better to design your model so the values
        # you want to see here are just included in the output, like
        # voltage is in this template model, for example.
        #
        # If you limit the number of batches you perform this operation
        # on, make sure you include enough info in your logger and
        # output files to associate the values with the right epoch and
        # step.
        # [!] 'Model' object has no attribute 'layers' : I think we can
        #     add in model a .layers attribute and just make it a list,
        #     because it will have shallow copies

        # [*] If there's any information you'd like to log
        # about individual layers, do so here.
        #
        # In this example, we're using a whitelist of layers to
        # log the below information for. If you wish to log the
        # information of all layers, move the `.log()` call
        # outside of this `if` guard.
        for layer in self.model.layers:
            # Log each of the weights defining the layer's state.
            #
            # For a linear layer, these weights are `w` and `b`.
            for i in range(len(layer.weights)):
                self.logger.log(
                    data_label=layer.name + '.w' + str(i),
                    data=layer.weights[i].numpy(),
                    meta={
                        'stride': 'step',

                        'description':
                            'w weights of ' + layer.name
                    }
                )

            # Log any losses associated with the layer
            self.logger.log(
                data_label=layer.name + '.losses',
                data=layer.losses,
                meta={
                    'stride': 'step',

                    'description':
                        'losses for ' + layer.name
                }
            )

            # [*] This is how you calculate layer outputs for
            # layers your network isn't directly reporting. This is
            # very expensive, so if you can have your network
            # report directly, or access this data anywhere besides
            # the training loop, that's preferable. Nevertheless,
            # should you wish to partake in the dark arts, here you
            # go:
            #
            # ```
            # kf = K.function([self.model.input], [layer.output])
            # self.logger.log(
            #     data_label=layer.name + '.outputs',
            #     data=kf([batch_x]),
            #     meta={
            #         'stride': 'step',
            #
            #         'description':
            #             'outputs for ' + layer.name
            #     }
            # )
            # ```

        # Log the calculated step loss
        self.logger.log(
            data_label='step_loss',
            data=float(loss),
            meta={
                'stride': 'step',

                'description':
                    'calculated step loss'
            }
        )
        self.logger.on_step_end()

        return loss  # in classification tasks, also return accuracy


    #┬───────────────────────────────────────────────────────────────────────╮
    #┤ Training Loop (epoch level)                                           │
    #┴───────────────────────────────────────────────────────────────────────╯

    def train_epoch(self, epoch_idx=None):
        """Train over an epoch.

        Also performs logging at the level of epoch metrics.
        """
        train_cfg = self.cfg['train']

        profile_epoch = (  # [!] duplicate code
            (epoch_idx + 1) in self.cfg['log'].profiler_epochs
            and self.cfg['log'].run_profiler
        )

        #┬───────────────────────────────────────────────────────────────────╮
        #┤ Epochwise Logging (pre-epoch)                                     │
        #┴───────────────────────────────────────────────────────────────────╯

        # [*] Declare epoch-level log variables (logged after training)
        losses = []

        #┬───────────────────────────────────────────────────────────────────╮
        #┤ Epochwise Training                                                │
        #┴───────────────────────────────────────────────────────────────────╯

        dataset = self.data.get()
        pb = Progbar(train_cfg.n_batch, stateful_metrics=None)

        # Iterate over training steps
        for step_idx in range(train_cfg.n_batch):
            # NOTE: trace events only created when profiler is enabled
            # (i.e. this isn't costly if the profiler is off)
            with profiler.Trace('train', step_num=step_idx, _r=1):
                # [!] implement range (i.e. just 1-10 batches)
                (batch_x, batch_y) = self.data.next()
                loss = self.train_step(batch_x, batch_y, step_idx)

            # Update progress bar
            pb.add(
                1,
                values=[
                    # [*] Register real-time epoch-level log variables.
                    # These are what show up to the right of the
                    # progress bar during training.
                    ('loss', loss),
                ]
            )

            # [*] Update epoch-level log variables
            losses.append(loss)

        #┬───────────────────────────────────────────────────────────────────╮
        #┤ Epochwise Logging (post-epoch)                                    │
        #┴───────────────────────────────────────────────────────────────────╯

        # [*] Post-training operations on epoch-level log variables
        epoch_loss = np.mean(losses)
        # [*] Log any epoch-wise variables.
        self.logger.log(
            data_label='epoch_loss',
            data=epoch_loss,
            meta={
                'stride': 'epoch',

                'description':
                    'mean step loss within an epoch'
            }
        )

        # [*] Summarize epoch-level log variables here
        # [?] Register epoch-level log variables here
        self.logger.summarize(
            epoch_idx,
            summary_items={
                ("epoch_loss", epoch_loss),
            }
        )

        return epoch_loss


    #┬───────────────────────────────────────────────────────────────────────╮
    #┤ Training Loop                                                         │
    #┴───────────────────────────────────────────────────────────────────────╯

    # [?] Should we be using @tf.function somewhere?
    # [!] Annoying how plt logging shows up
    def train(self):
        """TODO: docs"""
        n_epochs = self.cfg['train'].n_epochs

        #┬───────────────────────────────────────────────────────────────────╮
        #┤ Logging (pre-training)                                            │
        #┴───────────────────────────────────────────────────────────────────╯

        self.logger.on_train_begin();

        # Create checkpoint manager  # [?] move to logger?
        ckpt = tf.train.Checkpoint(
            step=tf.Variable(1),
            optimizer=self.optimizer,
            model=self.model
        )
        cpm = tf.train.CheckpointManager(
            ckpt,
            self.cfg['save'].checkpoint_dir,
            max_to_keep=3
        )

        #┬───────────────────────────────────────────────────────────────────╮
        #┤ Training                                                          │
        #┴───────────────────────────────────────────────────────────────────╯

        for epoch_idx in range(n_epochs):
            """[*] Other stuff you can log
            print("W = {}, B = {}".format(*self.model.trainable_variables))
            for k in self.model.trainable_variables:
                print("trainable_variables:")
                print(k)
            """
            action_list = self.logger.on_epoch_begin() # put profiler in here?

            profile_epoch = (
                self.cfg['log'].run_profiler
                and (epoch_idx + 1) in self.cfg['log'].profiler_epochs
            )

            # Start profiler
            if profile_epoch:
                # [!] implement range
                try:
                    profiler.start(self.cfg['save'].profile_dir)
                except Exception as e:
                    logging.warning(f"issue starting profiler: {e}")

            print(
                f"\nEpoch {epoch_idx + 1} / {n_epochs}"
                + f" (batch size = {self.cfg['train'].batch_size}):"
            )
            loss = self.train_epoch(epoch_idx)

            #┬───────────────────────────────────────────────────────────────╮
            #┤ Logging (mid-training)                                        │
            #┴───────────────────────────────────────────────────────────────╯

            # Save checkpoints
            # [?] move to logger
            # [!] not integrated with broader training paradigm
            # [!] still don't have full model saving
            if self.cfg['log'].ckpt_freq * self.cfg['log'].ckpt_lim > 0:
                ckpt.step.assign_add(1)
                if epoch_idx == 2:
                    save_path = cpm.save()

            # Logger-controlled actions (prefer doing things in the
            # logger when possible, use this when not)
            action_list = self.logger.on_epoch_end()
            """
            if 'save_weights' in action_list:
                # Create checkpoints
                tf.saved_model.save(self.model, self.cfg['save'].checkpoint_dir)
                self.model.save_model(os.path.join(
                    self.cfg['save'].checkpoint_dir,
                    f"checkpoint_e{epoch_idx + 1}"
                ))
            """

            # Stop profiler
            if profile_epoch:
                try:
                    profiler.stop()
                except Exception as e:
                    logging.warning(f"issue stopping profiler: {e}")
                # [!] implement range

        #┬───────────────────────────────────────────────────────────────────╮
        #┤ Logging (post-training)                                           │
        #┴───────────────────────────────────────────────────────────────────╯

        self.logger.on_train_end();
