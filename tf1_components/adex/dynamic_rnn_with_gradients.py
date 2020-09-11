# -*- coding: utf-8 -*-

import tensorflow as tf
from collections import Iterable
from tensorflow.python.util import nest


def namedtuple_to_list(state):
    if isinstance(state, list):
        return state
    return list(state)


def list_to_namedtuple(state_var_list, namedtuple_constructor):
    if namedtuple_constructor is None:
        return state_var_list
    return namedtuple_constructor(*state_var_list)


def tf_get_shape(tensor):
    """
    This function returns the shape of the tensor with all static components preserved.
    Returns a tuple with the shapes where only the unknown dimensions are
    replaced by the tensorflow shape tensor. This helps with stricter
    graph-generation time size checking
    """
    tensor_shape_tf = tf.shape(tensor)
    tensor_shape = list(tensor.shape)
    for i, arrdim in enumerate(tensor_shape):
        if arrdim.value is None:
            # replace unknown dimensions with tensorflow tensors
            tensor_shape[i] = tensor_shape_tf[i]
    return tuple(tensor_shape)


def dynamic_rnn_with_gradients(cell, inputs, sequence_lengths=None, dtype=tf.float32, initial_state=None,
                               swap_memory=False, return_states=False):
    def get_shape_elem(elem):
        if not isinstance(elem, Iterable):
            return (elem,)
        else:
            return tuple(elem)

    input_shape = tf_get_shape(inputs)
    batch_size = input_shape[0]
    n_time = input_shape[1]

    state_sizes_flat = nest.flatten(cell.state_size)
    output_sizes_flat = nest.flatten(cell.output_size)

    # initialize loop variables
    if initial_state is None:
        initial_state = cell.zero_state(batch_size, dtype=dtype)

    zero_output_flat = [tf.zeros((batch_size,) + get_shape_elem(size)) for size in output_sizes_flat]

    append_none = lambda ns: [None, ns] if isinstance(ns, int) else [None] + list(ns)
    output_arrays = [tf.TensorArray(dtype=dtype, size=n_time, dynamic_size=False, clear_after_read=False,
                                    element_shape=append_none(n)) for n in output_sizes_flat]
    state_arrays = [tf.TensorArray(dtype=dtype, size=n_time, dynamic_size=False, clear_after_read=False,
                                   element_shape=append_none(n)) for n in state_sizes_flat]

    # compute activity
    loop_vars = [0, output_arrays, state_arrays, initial_state]

    def rnn_single_step(t, state, inputs):
        with tf.name_scope('RNNSingleStep'):
            state_flat = list(state)
            actual_out, actual_state = cell(state=state, inputs=inputs)
            actual_out_flat = nest.flatten(actual_out)
            actual_state_flat = nest.flatten(actual_state)

            # Constrict the network in a manner similar to dynamic_rnn
            if sequence_lengths is not None:
                new_out_flat = [tf.where(tf.greater(sequence_lengths, t), newout, zeroout)
                                for newout, zeroout in zip(actual_out_flat, zero_output_flat)]
                new_state_flat = [tf.where(tf.greater(sequence_lengths, t), newstate, currstate)
                                  for newstate, currstate in zip(actual_state_flat, state_flat)]
            else:
                new_out_flat = [tf.identity(x) for x in actual_out_flat]
                new_state_flat = [tf.identity(x) for x in actual_state_flat]

            new_out = nest.pack_sequence_as(cell.output_size, new_out_flat)
            new_state = nest.pack_sequence_as(cell.state_size, new_state_flat)

        return new_out, new_state, actual_out, actual_state

    def loop_condition(t, outputs_arrays, state_arrays, state):
        return t < n_time

    def loop_body(t, output_arrays, state_arrays, state):
        new_state_arrays = [state_array.write(t, var) for state_array, var in zip(state_arrays, nest.flatten(state))]

        new_out, new_state, _, _ = rnn_single_step(t=t, state=state, inputs=inputs[:, t, :])
        new_output_arrays = [output_array.write(t, new_out_i) for output_array, new_out_i in
                                 zip(output_arrays, nest.flatten(new_out))]
        new_t = t + 1
        return new_t, new_output_arrays, new_state_arrays, new_state,

    loop_vars = tf.while_loop(loop_condition, loop_body, loop_vars, swap_memory=swap_memory, parallel_iterations=1)
    t_end, output_arrays, state_arrays, final_state = loop_vars

    # This is an assert to ensure that there is no dependency between the outputs and the final state
    # This is the case because the while loop appears to return an identity transformation of the final
    # state
    final_state_flat = nest.flatten(final_state)
    debuggrad = tf.gradients([output_array.read(n_time - 1) for output_array in output_arrays], final_state_flat)
    assert all(x is None for x in debuggrad)

    # Create outputs vector to return
    outputs_flat = [
        tf.transpose(output_array.stack(), perm=[1, 0] + list(range(2, 1 + len(output_array._element_shape[0]))))
        for output_array in output_arrays]
    # restore time dimension length that is made indeterminate when stacking the tensorarray
    outputs_flat = [tf.reshape(out, [-1, n_time] + out.shape[2:].as_list()) for out in outputs_flat]
    outputs = nest.pack_sequence_as(cell.output_size, outputs_flat)

    states_flat = [
        tf.transpose(state_array.stack(), perm=[1, 0] + list(range(2, 1 + len(state_array._element_shape[0]))))
        for state_array in state_arrays]
    # restore time dimension length that is made indeterminate when stacking the tensorarray
    states_flat = [tf.reshape(state, [-1, n_time] + state.shape[2:].as_list()) for state in states_flat]
    states = nest.pack_sequence_as(cell.state_size, states_flat)


    def gradient_function(loss, get_gradients_wrt_vars=False, name=None):
        """
        This function returns operations that calculate the gradient of the specified
        loss with respect to the state

        :returns: A named_tuple (of the same class used by the state) where each
            element is of size [batch_size, time_length, ..other element dims..] where
            each element gives the derivative of the loss with respect to this element
            in the state tuple in the corresponding batch at the corresponding time
        """

        with tf.name_scope('DynamicRNNGradientFunction'):
            with tf.name_scope('Tensor' if name is None else name):

                # here we use that the gradient is broken by calling stack. This
                # implies that this calculates only partial derivatives and not
                # derivatives that take into account the dependencies caused by the
                # network and intra-state dependencies
                # Moreover it appears that the while loop itself doesn't return the
                # original state but an identity transformation of it thus it is now
                # independent of outputs. Hence both of these derivatives are partial (i.e.
                # assuming that the other is constant)

                def zero_none_gradients(gradients, ref_tensors):
                    # replace None with zeros of appropriate size
                    return [gr if gr is not None else tf.zeros_like(ref_t, dtype=dtype)
                            for gr, ref_t in zip(gradients, ref_tensors)]

                def sum_gradient_lists(*args):
                    return [sum(x) for x in zip(*args)]

                de_doutput_partial = tf.gradients(loss, outputs_flat)
                de_doutput_partial = zero_none_gradients(de_doutput_partial, outputs_flat)

                dloss_dfinalstate_partial = tf.gradients(loss, final_state_flat)
                dloss_dfinalstate_partial = zero_none_gradients(dloss_dfinalstate_partial, final_state_flat)
                # This is the initial value of de_dnew_state (see the loop_body function
                # for further details)
                de_dnew_state_init = dloss_dfinalstate_partial

                def get_grad_vars():
                    if get_gradients_wrt_vars:
                        assert hasattr(cell, 'get_trainable_variables'), \
                            ("The RNNCell must implement the method get_trainable_variables in"
                             " order to calculate derivatives with respect to internal variables")
                        grad_var_dict = cell.get_trainable_variables()
                        grad_vname_used = grad_var_dict['used']
                        grad_vname_original = grad_var_dict['original']
                        grad_var_used = [getattr(cell, vname) for vname in grad_vname_used]
                        grad_var_original = [getattr(cell, vname) for vname in grad_vname_original]
                        grad_used_size_list = [x.shape for x in grad_var_used]
                        del grad_var_dict
                    else:
                        grad_vname_used = []
                        grad_vname_original = []
                        grad_var_used = []
                        grad_var_original = []
                        grad_used_size_list = []
                    return (grad_vname_used, grad_vname_original,
                            grad_var_used, grad_var_original,
                            grad_used_size_list)

                (grad_vname_used, grad_vname_original,
                 grad_var_used, grad_var_original,
                 grad_used_size_list) = get_grad_vars()
                gradient_arrays = [tf.TensorArray(dtype=dtype, size=n_time, clear_after_read=False,
                                                  element_shape=append_none(svarsize))
                                   for svarsize in state_sizes_flat]
                gradient_arrays.extend([tf.TensorArray(dtype=dtype, size=n_time, clear_after_read=False,
                                                       element_shape=varsize)
                                        for varsize in grad_used_size_list])  # NO batch dimension!

                loop_vars = [t_end - 1, gradient_arrays, de_dnew_state_init]

                def loop_condition(t, gradient_arrays, de_dnew_state):
                    return t >= 0

                def loop_body(t, gradient_arrays, de_dnew_state):
                    """
                    This is the graph

                    at time t:
                                                     ┌─────────────────┐
                                                ┌──> │ actual_state_t, │
                          trainable_variables ─────> │ actual_output_t │
                                                │    └────────┬────────┘
                                                │             │
                                                │             v
                    ... new_state_t_minus_1 ────┴──> check_sequence_len ─┬─> new_state_t  ... ... final_state
                                                              ʌ          │
                                                zero_output ──┘          │
                    ... new_output_t_minus_1                             └─> new_output_t ...

                    ┌──────────────────┐
                    │ new_output_0     │
                    │ new_output_1     │
                    │       .          ├────────┐
                    │       .          │        │   ┌──────┐
                    │ new_output_t_end │        ├──>│ loss │
                    └──────────────────┘        │   └──────┘
                         ┌─────────────┐        │
                         │ final_state ├────────┘
                         └─────────────┘
                    de_dnew_state is the gradient of loss with respect to new_state_t

                    The final gradient array[i] gives the gradient with respect to the
                    actual_state at time i (This is in accordance to the other static
                    simulation). This means that there is no gradient with respect to
                    the initialization state but there is one with respect to the final
                    state.
                    """

                    # de_dnew_state = [tf.Print(x, [t_end - t, x], "de_dstate_almost_{}: ".format(i), summarize=20)
                    #                     for i, x in enumerate(de_dnew_state)]
                    state_flat = [state_array.read(t) for state_array in state_arrays]
                    state = nest.pack_sequence_as(cell.state_size, state_flat)

                    # This is a bit of a hack to calculate gradients with respect to the variables used
                    # in the network. The basic problem is that these variables are created in the
                    # __init__ function of the cell i.e. outside the while context which we're
                    # currently in. Tensorflow does not allow backpropagating the gradients to such
                    # tensors. Therefore what we do is that we expect the
                    # cell.get_trainable_variables() method to give us two sets of variables:
                    #
                    # 1.  'used' variables: This list should contain ALL the class members that are
                    #     used in the calling function
                    # 2.  'original' variables: These must be tensors that are the variables with
                    #     respect to which you want to calculate the derivative.
                    #
                    # What we do is that we replace the member variables specified in 'used' with
                    # copies of their current tensors (created using tf.identity) before performing
                    # the call. These copies are now tensors that are created within the while
                    # context and thus we may calculate the gradient of the loss with respect to
                    # them. The derivative with respect to the variables in 'original' is
                    # calculated outside the while loop making use of the dependencies between the
                    # 'original' and 'used' variables

                    grad_var_list = []
                    if get_gradients_wrt_vars:
                        for vname, var in zip(grad_vname_used, grad_var_used):
                            var_copy = tf.identity(var)  # Create new tensor within while context and replace the tensor in cell
                            setattr(cell, vname, var_copy)  # This is necessary as you cannot back-propagate
                            grad_var_list.append(var_copy)  # to a tensor that was created outside the while context

                    new_output, new_state, actual_output, actual_state = rnn_single_step(t=t, state=state,
                                                                                         inputs=inputs[:, t, :])
                    state_flat = nest.flatten(state)
                    new_state_flat = nest.flatten(new_state)
                    new_output_flat = nest.flatten(new_output)
                    actual_state_flat = nest.flatten(actual_state)

                    # This is a list of ALL variables with respect to which we are interested in the gradient
                    all_gradient_vars = actual_state_flat + grad_var_list

                    de_doutput_partial_t = [(dout[:, t, :] if dout is not None else None)
                                            for dout in de_doutput_partial]
                    de_doutput_partial_t = zero_none_gradients(de_doutput_partial_t, new_output_flat)
                    de_dall_gradient_vars_new_output = tf.gradients(ys=new_output_flat, xs=all_gradient_vars,
                                                                    grad_ys=de_doutput_partial_t)
                    de_dall_gradient_vars_new_output = zero_none_gradients(de_dall_gradient_vars_new_output, all_gradient_vars)
                    de_dall_gradient_vars_new_state = tf.gradients(ys=new_state_flat, xs=all_gradient_vars,
                                                                   grad_ys=de_dnew_state)
                    de_dall_gradient_vars_new_state = zero_none_gradients(de_dall_gradient_vars_new_state, all_gradient_vars)

                    de_dall_gradient_vars = sum_gradient_lists(de_dall_gradient_vars_new_output,
                                                               de_dall_gradient_vars_new_state)
                    new_gradient_arrays = [ga.write(t, der) for der, ga in zip(de_dall_gradient_vars, gradient_arrays)]

                    if get_gradients_wrt_vars:
                        for vname, var in zip(grad_vname_used, grad_var_used):
                            setattr(cell, vname, var)  # Restore original tensors of cell object

                    # Here we calculate the de_dnew_state for the current state by taking
                    # the gradient as backpropagated through the new_output and new_state
                    de_dstate_new_output = tf.gradients(ys=new_output_flat, xs=state_flat, grad_ys=de_doutput_partial_t)
                    de_dstate_new_output = zero_none_gradients(de_dstate_new_output, new_state_flat)

                    de_dstate_new_state = tf.gradients(ys=new_state_flat, xs=state_flat, grad_ys=de_dnew_state)
                    de_dstate_new_state = zero_none_gradients(de_dstate_new_state, new_state_flat)

                    de_dnew_state = sum_gradient_lists(de_dstate_new_output, de_dstate_new_state)

                    new_t = t - 1
                    return new_t, new_gradient_arrays, de_dnew_state

                loop_vars = tf.while_loop(loop_condition, loop_body, loop_vars, swap_memory=swap_memory, name=str(name) + 'GradFunWhile', parallel_iterations=1)
                t, gradient_arrays, de_dstate = loop_vars

                # # writing the final gradient
                # de_dstate = [de_dstate_x + (dpart_x[:, t] if dpart_x is not None else 0) for (de_dstate_x, dpart_x)
                #              in zip(de_dstate, de_dstate_partial)]
                # gradient_arrays = [ga.write(t, der) for der, ga in zip(de_dstate, gradient_arrays)]

                gradient_arrays_wrt_state = gradient_arrays[:len(state_sizes_flat)]
                gradients_wrt_state = [tf.transpose(ga.stack(), perm=[1, 0] + list(range(2, 1 + len(ga._element_shape[0])))) for
                                       ga in gradient_arrays_wrt_state]
                gradients_wrt_state = [tf.reshape(gr, [-1, n_time] + gr.shape[2:].as_list()) for gr in gradients_wrt_state]
                gradients_wrt_state = nest.pack_sequence_as(cell.state_size, gradients_wrt_state)

                if not get_gradients_wrt_vars:
                    return gradients_wrt_state
                else:
                    gradient_arrays_wrt_used_vars = gradient_arrays[-len(grad_var_used):]
                    gradients_wrt_used_vars = [ga.stack() for ga in gradient_arrays_wrt_used_vars]
                    gradients_wrt_used_vars = [tf.reshape(gr, [n_time] + gr.shape[1:].as_list()) for gr in
                                               gradients_wrt_used_vars]

                    gradients_wrt_original_vars = tf.map_fn(lambda x: tf.gradients(grad_var_used, grad_var_original, x),
                                                            gradients_wrt_used_vars, dtype=[dtype] * len(grad_var_original))
                    gradients_wrt_original_vars = {vname: var for vname, var in zip(grad_vname_original,
                                                                                    gradients_wrt_original_vars)}
                    return gradients_wrt_state, gradients_wrt_original_vars

    if not(return_states):
        return outputs, final_state, gradient_function
    else:
        return outputs, final_state, gradient_function, states
