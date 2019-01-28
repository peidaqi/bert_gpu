# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Functions and classes related to optimization (weight updates)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import tensorflow as tf

from common import FLAGS
from tensorflow.python.training import distribution_strategy_context
from tensorflow.python.training.optimizer import get_filtered_grad_fn
from tensorflow.python.eager import context
from tensorflow.python.ops.gen_resource_variable_ops import resource_scatter_add
from tensorflow.contrib.optimizer_v2 import optimizer_v2


def create_optimizer(loss, init_lr, num_train_steps, num_warmup_steps, use_tpu):
    """Creates an optimizer training op."""
    global_step = tf.train.get_or_create_global_step()

    learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)

    # Implements linear decay of the learning rate.
    learning_rate = tf.train.polynomial_decay(
        learning_rate,
        global_step,
        num_train_steps,
        end_learning_rate=0.0,
        power=1.0,
        cycle=False)

    # Implements linear warmup. I.e., if global_step < num_warmup_steps, the
    # learning rate will be `global_step/num_warmup_steps * init_lr`.
    if num_warmup_steps:
        global_steps_int = tf.cast(global_step, tf.int32)
        warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

        global_steps_float = tf.cast(global_steps_int, tf.float32)
        warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

        warmup_percent_done = global_steps_float / warmup_steps_float
        warmup_learning_rate = init_lr * warmup_percent_done

        is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
        learning_rate = (
                (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)

        # It is recommended that you use this optimizer for fine tuning, since this
        # is how the model was trained (note that the Adam m/v variables are NOT
        # loaded from init_checkpoint.)

        #
        # Daqi: For now, let's cheat a bit by using a tf.contrib.distribute compatible optimizer
        #
        # if use_tpu:
        optimizer = AdamWeightDecayOptimizer2(
            learning_rate=learning_rate,
            weight_decay_rate=0.01,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-6,
            exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
    # else:
    #     optimizer = tf.train.AdamOptimizer(
    #         learning_rate=learning_rate,
    #         beta1=0.9,
    #         beta2=0.999,
    #         epsilon=1e-6
    #     )

    if use_tpu:
        optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

    tvars = tf.trainable_variables()
    #
    # Daqi - Strangely, it seems DistributionStrategy works even without
    # the distribution-aware Optimizer.compute_gradients.
    #
    grads = tf.gradients(loss, tvars)

    # This is how the model was pre-trained.
    (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)

    train_op = optimizer.apply_gradients(
        zip(grads, tvars), global_step=global_step)

    # Normally the global step update is done inside of `apply_gradients`.
    # However, `AdamWeightDecayOptimizer` doesn't do this. But if you use
    # a different optimizer, you should probably take this line out.
    new_global_step = global_step + 1
    train_op = tf.group(train_op, [global_step.assign(new_global_step)])
    return train_op

from tensorflow.contrib.optimizer_v2 import optimizer_v2
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import training_ops


class AdamWeightDecayOptimizer2(optimizer_v2.OptimizerV2):

    def __init__(self,
                 learning_rate=0.001,
                 weight_decay_rate=0.0,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-6,
                 exclude_from_weight_decay=None,
                 use_locking=False,
                 name='AdamWeightDecayOptimizer2'):
        super(AdamWeightDecayOptimizer2, self).__init__(use_locking, name)

        self._set_hyper('learning_rate', learning_rate)
        self._set_hyper('beta1', beta1)
        self._set_hyper('beta2', beta2)
        self._set_hyper('epsilon', epsilon)
        self._set_hyper('weight_decay_rate', weight_decay_rate)
        self.exclude_from_weight_decay = exclude_from_weight_decay

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        # if not self.weight_decay_rate:
        #     return False
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _get_variable_name(self, param_name):
        """Get the variable name from the tensor name."""
        m = re.match("^(.*):\\d+$", param_name)
        if m is not None:
            param_name = m.group(1)
        return param_name

    def _create_vars(self, var_list, state):
        # Create slots for the first and second moments.
        for v in var_list:
            state.zeros_slot(v, "m")
            state.zeros_slot(v, "v")

    def _apply_dense(self, grad, var, state):
        raise NotImplementedError()

    def _resource_apply_dense(self, grad, var, state):
        raise NotImplementedError()

    def _apply_sparse_shared(self, grad, var, indices, scatter_add, state):

        lr = state.get_hyper("learning_rate", var.dtype.base_dtype)
        beta1_t = state.get_hyper("beta1", var.dtype.base_dtype)
        beta2_t = state.get_hyper("beta2", var.dtype.base_dtype)
        epsilon_t = state.get_hyper("epsilon", var.dtype.base_dtype)
        weight_decay_rate = state.get_hyper('weight_decay_rate', var.dtype.base_dtype)
        # lr = (lr_t * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power))
        # m_t = beta1 * m + (1 - beta1) * g_t
        m = state.get_slot(var, "m")
        m_scaled_g_values = grad * (1 - beta1_t)
        m_t = state_ops.assign(m, m * beta1_t, use_locking=self._use_locking)
        with ops.control_dependencies([m_t]):
            m_t = scatter_add(m, indices, m_scaled_g_values)
        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v = state.get_slot(var, "v")
        v_scaled_g_values = (grad * grad) * (1 - beta2_t)
        v_t = state_ops.assign(v, v * beta2_t, use_locking=self._use_locking)
        with ops.control_dependencies([v_t]):
            v_t = scatter_add(v, indices, v_scaled_g_values)
        v_sqrt = math_ops.sqrt(v_t)
        # var_update = state_ops.assign_sub(
        #     var, lr * m_t / (v_sqrt + epsilon_t), use_locking=self._use_locking)
        var_update = m_t / (v_sqrt + epsilon_t)
        if self._do_use_weight_decay(self._get_variable_name(var.name)):
            var_update += weight_decay_rate * var
        update_with_lr = lr * var_update

        var_update = state_ops.assign_sub(
            var, update_with_lr, use_locking=self._use_locking)

        return control_flow_ops.group(*[var_update, m_t, v_t])

    def _apply_sparse(self, grad, var, state):
        return self._apply_sparse_shared(
            grad.values, var, grad.indices,
            lambda x, i, v: state_ops.scatter_add(  # pylint: disable=g-long-lambda
                x, i, v, use_locking=self._use_locking),
            state)

    def _resource_scatter_add(self, x, i, v):
        with ops.control_dependencies(
                [resource_variable_ops.resource_scatter_add(x.handle, i, v)]):
            return x.value()

    def _resource_apply_sparse(self, grad, var, indices, state):
        return self._apply_sparse_shared(grad, var, indices,
                                         self._resource_scatter_add, state)


class AdamWeightDecayOptimizer(tf.train.Optimizer):
    """A basic Adam optimizer that includes "correct" L2 weight decay."""

    def __init__(self,
                 learning_rate,
                 weight_decay_rate=0.0,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-6,
                 exclude_from_weight_decay=None,
                 name="AdamWeightDecayOptimizer"):
        """Constructs a AdamWeightDecayOptimizer."""
        super(AdamWeightDecayOptimizer, self).__init__(False, name)

        self.learning_rate = learning_rate
        self.weight_decay_rate = weight_decay_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.exclude_from_weight_decay = exclude_from_weight_decay

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """See base class."""

        assignments = []
        for (grad, param) in grads_and_vars:
            if grad is None or param is None:
                continue

            param_name = self._get_variable_name(param.name)

            m = tf.get_variable(
                name=param_name + "/adam_m",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer())
            v = tf.get_variable(
                name=param_name + "/adam_v",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer())

            # Standard Adam update.
            next_m = (
                    tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
            next_v = (
                    tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2,
                                                              tf.square(grad)))

            update = next_m / (tf.sqrt(next_v) + self.epsilon)

            # Just adding the square of the weights to the loss function is *not*
            # the correct way of using L2 regularization/weight decay with Adam,
            # since that will interact with the m and v parameters in strange ways.
            #
            # Instead we want ot decay the weights in a manner that doesn't interact
            # with the m/v parameters. This is equivalent to adding the square
            # of the weights to the loss with plain (non-momentum) SGD.
            if self._do_use_weight_decay(param_name):
                update += self.weight_decay_rate * param

            update_with_lr = self.learning_rate * update

            next_param = param - update_with_lr

            # assignments.extend(
            #     [param.assign(next_param),
            #      m.assign(next_m),
            #      v.assign(next_v)])
            # Daqi - this gives a small performance gain vs extend().
            assignments += [param.assign(next_param), m.assign(next_m), v.assign(next_v)]
        return tf.group(*assignments, name=name)

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if not self.weight_decay_rate:
            return False
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _get_variable_name(self, param_name):
        """Get the variable name from the tensor name."""
        m = re.match("^(.*):\\d+$", param_name)
        if m is not None:
            param_name = m.group(1)
        return param_name
