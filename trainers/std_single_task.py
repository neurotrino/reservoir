"""Sample of the two idiomatic ways to train.

tensorflow.org/tutorials/customization/custom_training_walkthrough
"""

# external ----
from tensorflow.keras.utils import Progbar

import logging
import numpy as np
import os
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.profiler.experimental as profiler

# internal ----
from models.common import fano_factor
from trainers.base import BaseTrainer
from utils.misc import SwitchedDecorator
from utils.connmat import (
    ExInOutputMatrixGenerator as OMG,
)

DEBUG_MODE = True

switched_tf_function = SwitchedDecorator(tf.function)
switched_tf_function.enabled = not DEBUG_MODE


class Trainer(BaseTrainer):
    """TODO: docs  | note how `optimizer` isn't in the parent"""

    def __init__(self, cfg, model, data, logger):
        super().__init__(cfg, model, data, logger)

        train_cfg = cfg["train"]

        try:
            if train_cfg.use_adam:
                self.main_optimizer = tf.keras.optimizers.Adam(
                    learning_rate=train_cfg.learning_rate
                )
                self.output_optimizer = tf.keras.optimizers.Adam(
                    learning_rate=train_cfg.output_learning_rate
                )
            else:
                self.main_optimizer = tf.keras.optimizers.SGD(
                    learning_rate=train_cfg.learning_rate
                )
                self.output_optimizer = tf.keras.optimizers.SGD(
                    learning_rate=train_cfg.output_learning_rate
                )
        except Exception as e:
            logging.warning(f"learning rate not set: {e}")

    # ┬───────────────────────────────────────────────────────────────────────╮
    # ┤ Training Loop (step level)                                            │
    # ┴───────────────────────────────────────────────────────────────────────╯


    @switched_tf_function
    def task_loss(self, x, y):
        return 0


    @switched_tf_function
    def rate_loss(self, spikes, target_rate, wgt=1.0, algorithm="simple"):

        # Compute the sum of squares difference between the target rate
        # and the observed rate over the entire timeseries
        if algorithm == "simple":
            unitwise_rates = tf.reduce_mean(spikes, axis=(0, 1))
            rate_loss = tf.reduce_sum(tf.square(unitwise_rates - target_rate))
            return rate_loss * wgt

        # Compute
        if algorithm == "rolling":

            # Convenience variables
            (s0, s1, s2) = spikes.shape

            # Reshape for convenience
            rs_spikes = tf.reshape(spikes, (s0 * s1, s2))

            # Compute running average
            denominators = tf.range(1, s0 * s1 + 1, dtype=tf.float32)[:, None]
            rolling_avg = tf.cumsum(rs_spikes) / denominators

            # Restore shape
            rolling_avg = tf.reshape(rolling_avg, spikes.shape)

            # Weighted rate loss
            rate_loss = tf.reduce_sum(
                tf.reduce_mean(
                    tf.square(rolling_avg - target_rate),
                    axis=1
                )
            )
            return rate_loss * wgt

        # Compute the rate loss using a window to compute the
        # instantaneous firing rate
        if algorithm == "windowed":
            raise NotImplementedError(
                "windowed rate loss is not currently supported"
            )


    @switched_tf_function
    def sync_loss(self, x, y):
        return 0


    @switched_tf_function
    def loss(self, x, y):
        """Calculate the loss on data x labeled y."""

        # Model output
        model_output = self.model(x)  # tripartite output
        voltage, spikes, prediction = model_output

        if self.cfg["model"].cell.categorical_output:
            cat_prediction = tf.math.divide_no_nan(
                prediction[:, :, 0], prediction[:, :, 1]
            )
            # cat_prediction = tf.scatter_nd(tf.math.is_nan(cat_prediction), 0, tf.shape(cat_prediction))
        # turns into a ratio
        # if >1, then one output has more activity
        # if <1, then the other output has more activity
        # targets are 0.5 (1/2) and 2.0 (2/1) for high and low coherences

        # Penalty for performance
        loss_object = tf.keras.losses.MeanSquaredError()
        if self.cfg["model"].cell.categorical_output:
            task_loss = loss_object(y_true=y, y_pred=cat_prediction)
        else:
            task_loss = loss_object(y_true=y, y_pred=prediction)

        if not self.cfg['train'].include_task_loss:
            task_loss = task_loss * 0.0

        net_loss = task_loss

        # Penalty for unrealistic firing rates
        # [!] need to add metric or something else for real-time
        #     reporting of loss components (could store all losses in
        #     a model attribute and add to our loading bar plus write
        #     to disk then flush)
        if self.cfg["train"].simple_rate_loss:
            rate_loss = self.rate_loss(
                spikes,
                self.cfg["train"].target_rate,
                self.cfg["train"].rate_cost,
                "simple"
            )
        else:
            rate_loss = self.rate_loss(
                spikes,
                self.cfg["train"].target_rate,
                self.cfg["train"].rate_cost,
                "rolling"
            )

        if self.cfg["train"].lax_rate_loss:
            if rate_loss < task_loss * self.cfg["train"].lax_rate_threshold:
                rate_loss = 0.0

        if self.cfg["train"].include_rate_loss:
            net_loss = net_loss + rate_loss

        # Penalty for unrealistic synchrony
        synchrony = fano_factor(self, self.cfg["data"].seq_len, spikes)
        synch_loss = (
            tf.reduce_sum(
                tf.square(synchrony - self.cfg["train"].target_synch)
            )
            * self.cfg["train"].synch_cost
        )
        if self.cfg["train"].lax_synch_loss:
            if synch_loss < task_loss * self.cfg["train"].lax_synch_threshold:
                synch_loss = 0.0
        if self.cfg["train"].include_synch_loss:
            net_loss += synch_loss

        # Feed model output and losses back up the stack
        # [!] would like to make loss scalars a Keras `metric` but we
        #     can't do that unless we refactor layers as in branch 68
        losses = (
            task_loss,
            rate_loss,
            synch_loss,
            net_loss,
        )
        return (model_output, losses)

    @switched_tf_function
    def grad(self, inputs, targets):
        """Gradient calculation(s)"""
        with tf.GradientTape() as tape:
            (model_output, losses) = self.loss(inputs, targets)

        # Calculate the gradient of the loss with respect to each
        # layer's trainable variables. In this example, calculates the
        # gradients for (in order):
        # > `rnn/ex_in_lif/input_weights:0`
        # > `rnn/ex_in_lif/recurrent_weights:0`
        # > `dense/kernel:0`
        # > `dense/bias:0`

        if self.cfg["train"].noise_weights_before_gradient:
            self.model.noise_weights()

        """
        if self.cfg["train"].include_rate_loss and self.cfg["train"].include_task_loss:
            grads = tape.gradient(losses[-1], self.model.trainable_variables)
        elif self.cfg["train"].include_rate_loss and not self.cfg["train"].include_task_loss:
            grads = tape.gradient(losses[1], self.model.trainable_variables)
        elif self.cfg["train"].include_task_loss and not self.cfg["train"].include_rate_loss:
            grads = tape.gradient(losses[0], self.model.trainable_variables)
        """
        grads = tape.gradient(losses[-1], self.model.trainable_variables)
        return (model_output, losses, grads)

    # @switched_tf_function  # [!] might need this
    def train_step(self, batch_x, batch_y, batch_idx=None):
        """Train on the next batch."""

        # ┬───────────────────────────────────────────────────────────────────╮
        # ┤ Pre-Step Logging                                                  │
        # ┴───────────────────────────────────────────────────────────────────╯

        self.logger.on_step_begin()

        # Input/reference variables
        self.logger.log(
            data_label="inputs",
            data=batch_x.numpy(),
            meta={
                "stride": "step",
                "shape_key": (
                    "num_batches * post_every",
                    "batch_size",
                    "seq_len",
                    "n_input",
                ),
                "description": "inputs",
            },
        )
        self.logger.log(
            data_label="true_y",
            data=batch_y.numpy(),
            meta={
                "stride": "step",
                "shape_key": (
                    "num_batches * post_every",
                    "batch_size",
                    "seq_len",
                    "1",
                ),
                "description": "correct values",
            },
        )

        # [!] empty first time (needs at least one forward pass)
        # [!] besides the first time, this is just postweights of the
        #     last batch, so we want to have a `static` save of the
        #     first time, but not this
        # preweights = [x for x in self.model.trainable_variables]

        # ┬───────────────────────────────────────────────────────────────────╮
        # ┤ Gradient Calculation                                              │
        # ┴───────────────────────────────────────────────────────────────────╯

        (model_output, losses, grads) = self.grad(batch_x, batch_y)
        voltage, spikes, prediction = model_output

        (task_loss, rate_loss, synch_loss, net_loss) = losses

        # declare layer-wise vars and grads so that layer-wise optimization can occur
        self.var_list1 = self.model.rnn1.trainable_variables
        self.var_list2 = self.model.dense1.trainable_variables
        grads1 = grads[: len(self.var_list1)]
        grads2 = grads[len(self.var_list1) :]

        # ┬───────────────────────────────────────────────────────────────────╮
        # ┤ Mid-Step Logging                                                  │
        # ┴───────────────────────────────────────────────────────────────────╯

        # Output variables
        self.logger.log(
            data_label="voltage",
            data=voltage.numpy(),
            meta={
                "stride": "step",
                "shape_key": (
                    "num_batches * post_every",
                    "batch_size",
                    "seq_len",
                    "n_recurrent",
                ),
                "description": "voltages",
            },
        )
        self.logger.log(
            data_label="spikes",
            data=spikes.numpy(),
            meta={
                "stride": "step",
                "shape_key": (
                    "num_batches * post_every",
                    "batch_size",
                    "seq_len",
                    "n_recurrent",
                ),
                "description": "spikes",
            },
        )
        self.logger.log(
            data_label="pred_y",
            data=prediction.numpy(),
            meta={
                "stride": "step",
                "shape_key": (
                    "num_batches * post_every",
                    "batch_size",
                    "seq_len",
                    "1",
                ),
                "description": "predictions",
            },
        )

        # ┬───────────────────────────────────────────────────────────────────╮
        # ┤ Gradient Application                                              │
        # ┴───────────────────────────────────────────────────────────────────╯
        """
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables)
        )
        """

        if self.model.training:
            self.main_optimizer.apply_gradients(
                zip(grads1, self.model.rnn1.trainable_variables)
            )
            self.output_optimizer.apply_gradients(
                zip(grads2, self.model.dense1.trainable_variables)
            )

        if self.cfg["train"].noise_weights_after_gradient and self.model.training:
            self.model.noise_weights()

        # ┬───────────────────────────────────────────────────────────────────╮
        # ┤ Sparsity Enforcement                                              │
        # ┴───────────────────────────────────────────────────────────────────╯

        # if freewiring is permitted (i.e. sparsity of any kind is NOT enforced),
        # this step is needed to explicitly ensure self-recurrent
        # connections remain zero otherwise (if sparsity IS enforced)
        # that is taken care of through the rec_sign application below
        if self.cfg["model"].cell.freewiring:  # TODO: document in HJSON
            # Make sure all self-connections remain 0
            self.model.cell.recurrent_weights.assign(
                tf.where(
                    self.model.cell.disconnect_mask,
                    tf.zeros_like(self.model.cell.recurrent_weights),
                    self.model.cell.recurrent_weights,
                )
            )

        # If the sign of a weight changed from the original or the
        # weight (previously 0) is no longer 0, make the weight 0.
        #
        # Reminder that rec_sign contains 0's for initial 0's when
        # rewiring = true whereas it contains +1's or -1's (for excit
        # or inhib) for initial 0's when rewiring = false (freewiring = true)
        self.model.cell.recurrent_weights.assign(
            tf.where(
                self.model.cell.rec_sign * self.model.cell.recurrent_weights
                > 0,
                self.model.cell.recurrent_weights,
                0,
            )
        )

        # THIS SECTION parallels how sparsity is maintained for the RSNN even in the absence of rewiring
        # If the sign of an input or output weight changed from the original (or updated from last step)
        # or the weight is no longer 0, make the weight 0.
        # output_sign and input_sign contains 0's for former 0's, +1 for former positives, and -1 for former negatives.
        #
        self.model.dense1.oweights.assign(
            tf.where(
                self.model.dense1.output_sign * self.model.dense1.oweights
                > 0,
                self.model.dense1.oweights,
                0,
            )
        )

        # maintain sparsity of input weights (whether rewiring or not)
        if self.cfg["model"].cell.specify_input:
            self.model.cell.input_weights.assign(
                tf.where(
                    self.model.cell.input_sign * self.model.cell.input_weights > 0,
                    self.model.cell.input_weights,
                    0,
                )
            )

        # [!] prefer to have something like "for cell in model.cells,
        #     if cell.rewiring then cell.rewire" to better generalize;
        #     would involve adding a 'cells' attribute to model
        if self.model.cell.rewiring:
            self.model.cell.rewire()
            # correction to note as of Oct 8, 2021:
            # else statement removed (reverted to original sequence of execution)
            # all that needs to happen to circumvent the identified issue below
            # is for input_sign and rec_sign to be updated according within rewire.
            # the above (outside if statement) is to eliminate new weights (set to 0)
            # the within (rewire) takes care of new zeros (true zeros and also sign flips).
            # rec_sign and input_sign are updated to reflect new connections that
            # replace the zeroed ones. this prevents overwriting new connections
            # upon the next application of sparsity enforcement (above outside if statement).

            # note as of Oct 8, 2021:
            # the following has been moved inside a new else statement
            # meaning it only executes if rewiring = false
            # i believe previously, when placed outside an if/else statement
            # and in front of the if rewiring statement, the execution of
            # the application of rec_sign (according to initial zeros) meant
            # that rewiring was still not occuring. all changed / rewired zeros
            # get immediately changed back in the next update, even if they are
            # filled in here. the steps would've been:
            # 1. update weights (may include new values and new zeros)
            # 2. all new values in initial zero spots go to zero according to below script
            # 3. all new zeros get a value in rewire()
            # 4. loops through again in next update, thus canceling out all new zeros
            # Thus this script now ONLY happens if rewiring = false (initial zeros must stay zero)

        # rewire output weights
        if (
            self.cfg["train"].output_trainable
            and self.cfg["model"].cell.output_rewiring
        ):
            self.model.dense1.rewire()
        elif (
            self.cfg["train"].output_trainable
            and self.cfg["model"].cell.output_rewiring
            and self.cfg["model"].cell.specify_input
            and self.cfg["model"].cell.no_input_to_output
        ):
            self.model.dense1.rewire(self.model.cell.input_id)

        # rewire input weights
        # you can imagine not wanting to rewire input weights, especially if you
        # would want to maintain separation between input and output throughout
        # the course of training; easier to simply not rewire the input.
        # however then we might get sparser and sparser, and we don't want that either.
        if (
            self.cfg["train"].input_trainable
            and self.cfg["model"].cell.specify_input
            and self.cfg["model"].cell.input_rewiring
        ):
            self.model.cell.input_rewire()

        # ┬───────────────────────────────────────────────────────────────────╮
        # ┤ Post-Step Logging                                                 │
        # ┴───────────────────────────────────────────────────────────────────╯

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
                data_label=tvar.name + ".shapes",
                data=tvar.shape,
                meta={
                    "stride": "static",
                    "description": "shape of layer " + tvar.name,
                },
            )

            # Calculated gradients
            self.logger.log(
                data_label=f"tv{i}.gradients",
                data=grads[i].numpy(),
                meta={
                    "stride": "step",
                    "description": "gradients calculated for " + tvar.name,
                },
            )

            """
            # Weights before applying gradients
            try:
                self.logger.log(
                    data_label=tvar.name + '.preweights',
                    data=preweights[i].numpy(),
                    meta={
                        'stride': 'static',
                        'description':
                            'layer weights for '
                            + tvar.name
                            + ' before applying the gradients'
                    }
                )
            except Exception as e:
                logging.warning(f"issue logging preweights: {e}")"""

            # Weights after applying gradients
            try:
                self.logger.log(
                    data_label=f"tv{i}.postweights",
                    data=tvar.numpy(),
                    meta={
                        "stride": "step",
                        "description": "layer weights for "
                        + tvar.name
                        + " after applying the gradients",
                    },
                )
            except:  # [!] prefer not to have try/except
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
                    data_label=layer.name + ".w" + str(i),
                    data=layer.weights[i].numpy(),
                    meta={
                        "stride": "step",
                        "description": "w weights of " + layer.name,
                    },
                )

            # Log any losses associated with the layer
            self.logger.log(
                data_label=layer.name + ".losses",
                data=layer.losses,
                meta={
                    "stride": "step",
                    "description": "losses for " + layer.name,
                },
            )

        # Log the calculated step loss
        self.logger.log(
            data_label="step_task_loss",
            data=float(task_loss),
            meta={
                "stride": "step",
                "description": "calculated step loss (task)",
            },
        )

        self.logger.log(
            data_label="step_rate_loss",
            data=float(rate_loss),
            meta={
                "stride": "step",
                "description": "calculated step loss (rate)",
            },
        )
        self.logger.log(
            data_label="step_synch_loss",
            data=float(synch_loss),
            meta={
                "stride": "step",
                "description": "calculated step loss (synchrony)",
            },
        )

        self.logger.log(
            data_label="step_loss",
            data=float(net_loss),
            meta={"stride": "step", "description": "calculated step loss"},
        )
        self.logger.on_step_end()

        # Log the target zeros and signs for input and recurrent layers
        self.logger.log(
            data_label="target_zcount",
            data=self.model.cell._target_zcount,
            meta={
                "stride": "step",
                "description": "target count of zeros for recurrent layer",
            },
        )

        self.logger.log(
            data_label="rec_sign",
            data=self.model.cell.rec_sign.numpy(),
            meta={
                "stride": "step",
                "description": "signs (pos/neg/zero) of recurrent layer",
            },
        )

        # logging input weights
        self.logger.log(
            data_label="input_w",
            data=self.model.cell.input_weights.numpy(),
            meta={"stride": "step", "description": "input layer weights"},
        )

        # logging output weights
        self.logger.log(
            data_label="output_w",
            data=np.array(self.model.dense1.get_weights()),
            meta={"stride": "step", "description": "output layer weights"},
        )

        return losses  # in classification tasks, also return accuracy

    # ┬───────────────────────────────────────────────────────────────────────╮
    # ┤ Training Loop (epoch level)                                           │
    # ┴───────────────────────────────────────────────────────────────────────╯

    def train_epoch(self, epoch_idx=None):
        """Train over an epoch.

        Also performs logging at the level of epoch metrics.
        """
        train_cfg = self.cfg["train"]

        profile_epoch = (epoch_idx + 1) in self.cfg[  # [!] duplicate code
            "log"
        ].profiler_epochs and self.cfg["log"].run_profiler

        # ┬───────────────────────────────────────────────────────────────────╮
        # ┤ Epochwise Logging (pre-epoch)                                     │
        # ┴───────────────────────────────────────────────────────────────────╯

        # [*] Declare epoch-level log variables (logged after training)
        net_losses = []
        task_losses = []
        rate_losses = []
        synch_losses = []

        # ┬───────────────────────────────────────────────────────────────────╮
        # ┤ Epochwise Training                                                │
        # ┴───────────────────────────────────────────────────────────────────╯

        pb = Progbar(train_cfg.n_batch, stateful_metrics=None)

        # Iterate over training steps
        for step_idx in range(train_cfg.n_batch):
            """
            # NOTE: trace events only created when profiler is enabled
            # (i.e. this isn't costly if the profiler is off)
            """
            with profiler.Trace("train", step_num=step_idx, _r=1):
                # [!] implement range (i.e. just 1-10 batches)
                (batch_x_rates, batch_y) = self.data.next()
                # generate Poisson spikes from rates
                random_matrix = np.random.rand(
                    batch_x_rates.shape[0],
                    batch_x_rates.shape[1],
                    batch_x_rates.shape[2],
                )
                # batch_x_spikes = (batch_x_rates - random_matrix > 0)*1.
                batch_x_spikes = tf.where(
                    (batch_x_rates - random_matrix > 0), 1.0, 0.0
                )
                (
                    task_loss,
                    rate_loss,
                    synch_loss,
                    net_loss,
                ) = self.train_step(batch_x_spikes, batch_y, step_idx)

            # Update progress bar
            pb.add(
                1,
                values=[
                    # [*] Register real-time epoch-level log variables.
                    # These are what show up to the right of the
                    # progress bar during training.
                    ("net loss", net_loss),
                    ("task loss", task_loss),
                    ("rate loss", rate_loss),
                    ("synch loss", synch_loss),
                ],
            )

            # [*] Update epoch-level log variables
            net_losses.append(net_loss)
            task_losses.append(task_loss)
            rate_losses.append(rate_loss)
            synch_losses.append(synch_loss)

        # ┬───────────────────────────────────────────────────────────────────╮
        # ┤ Epochwise Logging (post-epoch)                                    │
        # ┴───────────────────────────────────────────────────────────────────╯

        # [*] Post-training operations on epoch-level log variables
        epoch_loss = np.mean(net_losses)
        # [*] Log any epoch-wise variables.
        self.logger.log(
            data_label="epoch_loss",
            data=epoch_loss,
            meta={
                "stride": "epoch",
                "description": "mean step loss within an epoch",
            },
        )
        epoch_rate_loss = np.mean(rate_losses)
        # [*] Log any epoch-wise variables.
        self.logger.log(
            data_label="epoch_rate_loss",
            data=epoch_rate_loss,
            meta={
                "stride": "epoch",
                "description": "mean step loss within an epoch (rate)",
            },
        )
        epoch_synch_loss = np.mean(synch_losses)
        # [*] Log any epoch-wise variables.
        self.logger.log(
            data_label="epoch_synch_loss",
            data=epoch_synch_loss,
            meta={
                "stride": "epoch",
                "description": "mean step loss within an epoch (synchrony)",
            },
        )
        epoch_task_loss = np.mean(task_losses)
        # [*] Log any epoch-wise variables.
        self.logger.log(
            data_label="epoch_task_loss",
            data=epoch_task_loss,
            meta={
                "stride": "epoch",
                "description": "mean step loss within an epoch (task)",
            },
        )

        # [*] Summarize epoch-level log variables here
        # [?] Register epoch-level log variables here
        self.logger.summarize(
            epoch_idx,
            summary_items={
                ("epoch_loss", epoch_loss),
                ("epoch_task_loss", epoch_task_loss),
                ("epoch_rate_loss", epoch_rate_loss),
            },
        )

        return epoch_loss

    # ┬───────────────────────────────────────────────────────────────────────╮
    # ┤ Training Loop                                                         │
    # ┴───────────────────────────────────────────────────────────────────────╯

    # [?] Should we be using @switched_tf_function somewhere?
    # [!] Annoying how plt logging shows up
    def train(self):
        """TODO: docs"""
        n_epochs = self.cfg["train"].n_epochs

        # ┬───────────────────────────────────────────────────────────────────╮
        # ┤ Logging (pre-training)                                            │
        # ┴───────────────────────────────────────────────────────────────────╯

        self.logger.on_train_begin()

        # Create checkpoint manager  # [?] move to logger?
        """
        ckpt = tf.train.Checkpoint(
            step=tf.Variable(1),
            optimizer=self.optimizer,
            model=self.model
        )
        cpm = tf.train.CheckpointManager(
            ckpt,
            self.cfg['save'].checkpoint_dir,
            max_to_keep=3
        )"""

        # ┬───────────────────────────────────────────────────────────────────╮
        # ┤ Training                                                          │
        # ┴───────────────────────────────────────────────────────────────────╯

        def sampled_vector(size):
            return np.random.uniform(low=0.0, high=0.4, size=size)

        def zero_speckled_vector(size, fraction_zero):
            v = sampled_vector(size)
            zero_indices = np.random.choice(
                np.arange(v.size),
                size=int(v.size * fraction_zero),
                replace=False
            )
            v[zero_indices] = 0
            return v

        input_shape = (self.cfg["data"].seq_len, self.cfg["data"].n_input)

        for epoch_idx in range(n_epochs):
            """[*] Other stuff you can log
            print("W = {}, B = {}".format(*self.model.trainable_variables))
            for k in self.model.trainable_variables:
                print("trainable_variables:")
                print(k)
            """
            cell = self.model.cell

            # [MAKE SURE NO OVERLAP]
            # (~initializer)
            if epoch_idx == 0 and self.cfg["model"].cell.no_input_to_output:
                # this is the correct initialization of recurrent weights and
                # so on
                if self.cfg["model"].cell.two_input_populations_by_rate:
                    # separate based on greater rate of responses for one
                    # coherence level vs the other
                    coh1_pop = [ 0,  1,  2,  3,  8,  9, 10, 15]
                    coh0_pop = [ 4,  5,  6,  7, 11, 12, 13, 14]

                    # get output weights
                    self.model.dense1.build(input_shape=input_shape)
                    output_vals = self.model.dense1.oweights.numpy()
                    self.model.out_id = np.where(output_vals != 0)[0]

                    self.model.avail_id = np.setdiff1d(
                        np.array(range(0,300)), self.model.out_id
                    )
                    # randomly split into remaining available id's for the two
                    # coherence levels
                    self.model.avail_id_coh0 = np.random.choice(
                        self.model.avail_id,
                        size=(len(self.model.avail_id) // 2),
                        replace=False
                    )
                    self.model.avail_id_coh1 = np.setdiff1d(
                        self.model.avail_id, self.model.avail_id_coh0
                    )

                    # redraw input weights from the two coherence populations
                    # according to leftover values (avail_ids)
                    input_weights_val = np.zeros(shape=(cell.n_in, cell.units))

                    # determine how many values we need to fill in per coherence level
                    # conns_per_input = p_input * n_units
                    for inp_idx in range(cell.n_in):
                        if inp_idx in coh0_pop:
                            cohX = self.model.avail_id_coh0
                        else:
                            cohX = self.model.avail_id_coh1
                        input_weights_val[inp_idx][cohX] = sampled_vector(cohX.shape)

                    """
                    # former way of doing it, when the non-overlap was not
                    # explicitly specified upon init

                    in_pop_size = self.model.n_in // 2
                    e_rec_pop_size = self.model.num_ex // 2
                    i_rec_pop_size = self.model.num_in // 2
                    input_weights_val = np.zeros(shape=(self.model.n_in, self.model.units))

                    for i in range(0,self.model.n_in):
                        fraction_zero = 1 - self.cfg["cell"].p_input

                        # sample vector of 120 to-excitatory input weights
                        # include zeros proportionally, randomly
                        e_sample_input_vals = zero_speckled_vector(
                            size=(e_rec_pop_size,), fraction_zero=fraction_zero
                        )

                        # sample vector of 30 to-inhibitory input weights
                        # include zeros proportionally, randomly
                        i_sample_input_vals = zero_speckled_vector(
                            size=(i_rec_pop_size,), fraction_zero=fraction_zero
                        )

                        idx0 = e_rec_pop_size
                        idx1 = idx0 + e_rec_pop_size
                        idx2 = idx1 + i_rec_pop_size
                        idx3 = idx2 + i_rec_pop_size

                        if i in coh1_pop:
                            # second half of e
                            input_weights_val[i][idx0:idx1] = e_sample_input_vals
                            # first half of i
                            input_weights_val[i][idx1:idx2] = i_sample_input_vals
                        else:
                            # first half of e
                            input_weights_val[i][:idx0] = e_sample_input_vals
                            # second half of i
                            input_weights_val[i][idx2:idx3] = i_sample_input_vals
                    """

                    cell.build(input_shape=input_shape)
                    cell.input_weights.assign(input_weights_val)

                    # update the permissible zero indices and indices for rewiring
                    # get which units actually receive input
                    self.model.input_id = np.unique(
                        np.where(input_weights_val != 0)[1]
                    )
                    # store input weights' signs, where 0s are 0s
                    self.model.input_sign = tf.sign(cell.input_weights)

                    self.model.input_target_zcount = len(tf.where(cell.input_weights == 0))

                # save (i.e. overwrite) the true initial input weight matrix
                out_dir = self.cfg["save"].main_output_dir
                np.save(
                    os.path.join(out_dir, "input_preweights.npy"),
                    cell.input_weights.numpy(),
                )

            # [REDRAW OUTPUT]
            # if we are in the 50th-from-last epoch, redraw output weights entirely and keep training
            # (~constraint)
            if self.cfg["train"].redraw_output and (epoch_idx == n_epochs - 100):
                # redraw output from initial distribution
                # everything else about the network remains the same
                # output_vals = self.model.dense1.oweights.numpy()
                output_connmat_generator = OMG(
                    n_excit=self.model.cell.num_ex,
                    n_inhib=self.model.cell.num_in,
                    n_out=self.model.n_out,
                    p_from_e=self.cfg["model"].cell.p_eo,
                    p_from_i=self.cfg["model"].cell.p_io,
                    mu=self.model.cell.mu,
                    sigma=self.model.cell.sigma
                )
                new_oweights = output_connmat_generator.run_generator()
                self.model.dense1.oweights.assign(
                    new_oweights * self.cfg["model"].cell.output_multiplier
                )
                # update output sign and target zeros potentially
                self.model.dense1.output_sign = tf.sign(self.model.dense1.oweights)
                self.model.dense1.output_target_zcount = len(tf.where(
                    self.model.dense1.oweights == 0
                ))

            # [SILENCING]
            # if we are at the last epoch (1001th), silence parts accordingly
            # permit update still for now, since that's a headache to detangle,
            # but what we'll be looking for are the spike statistics and the loss
            # pre-update within that first batch, so it is alright. the rapid
            # ability to bounce back, if that happens, will be informative anyway.
            if epoch_idx==n_epochs-1:
                # determine the recurrent units that project to output
                input_vals = self.model.cell.input_weights.numpy()
                #output_vals = np.array(self.model.dense1.get_weights())
                output_vals = self.model.dense1.oweights.numpy()
                not_id = np.where(output_vals==0)[0]
                out_id = np.where(output_vals!=0)[0]
                if self.cfg["train"].matched_silencing:
                    # randomly select the same number of not_id as out_id units for silencing
                    not_id = np.random.choice(not_id, size=len(out_id), replace=False)

                # silencing inputs onto the units
                if self.cfg["train"].silence_input_to_nonproj:
                    input_vals[:,not_id] = 0
                    self.model.cell.input_weights.assign(input_vals)

                elif self.cfg["train"].silence_input_to_proj:
                    input_vals[:,out_id] = 0
                    self.model.cell.input_weights.assign(input_vals)

                # now for silencing the units themselves (recurrently, not to output)
                elif self.cfg["train"].silence_nonproj:
                    rec_vals = self.model.cell.recurrent_weights.numpy()
                    rec_vals[not_id,:] = 0
                    self.model.cell.recurrent_weights.assign(rec_vals)

                elif self.cfg["train"].silence_proj:
                    rec_vals = self.model.cell.recurrent_weights.numpy()
                    rec_vals[out_id,:] = 0
                    self.model.cell.recurrent_weights.assign(rec_vals)

            action_list = (
                self.logger.on_epoch_begin()
            )  # put profiler in here?

            profile_epoch = (
                self.cfg["log"].run_profiler
                and (epoch_idx + 1) in self.cfg["log"].profiler_epochs
            )

            # Start profiler
            if profile_epoch:
                # [!] implement range
                try:
                    profiler.start(self.cfg["save"].profile_dir)
                except Exception as e:
                    logging.warning(f"issue starting profiler: {e}")

            print(
                f"\nEpoch {epoch_idx + 1} / {n_epochs}"
                + f" (batch size = {self.cfg['train'].batch_size}):"
            )
            loss = self.train_epoch(epoch_idx)

            # ┬───────────────────────────────────────────────────────────────╮
            # ┤ Logging (mid-training)                                        │
            # ┴───────────────────────────────────────────────────────────────╯

            # Save checkpoints
            # [?] move to logger
            # [!] not integrated with broader training paradigm
            # [!] still don't have full model saving
            if self.cfg["log"].ckpt_freq * self.cfg["log"].ckpt_lim > 0:
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

        # ┬───────────────────────────────────────────────────────────────────╮
        # ┤ Logging (post-training)                                           │
        # ┴───────────────────────────────────────────────────────────────────╯

        self.logger.on_train_end()
