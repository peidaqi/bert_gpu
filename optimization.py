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
import re
import tensorflow as tf

from common import FLAGS
from tensorflow.python.training import distribution_strategy_context
from tensorflow.python.ops import resource_variable_ops
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
    if use_tpu:
        optimizer = AdamWeightDecayOptimizer(
            learning_rate=learning_rate,
            weight_decay_rate=0.01,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-6,
            exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
    else:
        optimizer = AdamWeightDecayOptimizer3(
            learning_rate=learning_rate,
            weight_decay_rate=0.01,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-6,
            exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
        #  optimizer = tf.train.AdamOptimizer(
        #  learning_rate=learning_rate,
        #  beta1=0.9,
        #  beta2=0.999,
        #  epsilon=1e-6
        #  )

    if use_tpu:
        optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

    tvars = tf.trainable_variables()
    grads = tf.gradients(loss, tvars)

    # This is how the model was pre-trained.
    (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)

    train_op = optimizer.apply_gradients(
        zip(grads, tvars), global_step=None) # We don't pass global_step here since it's updated using a separate op.

    # Normally the global step update is done inside of `apply_gradients`.
    # However, `AdamWeightDecayOptimizer` doesn't do this. But if you use
    # a different optimizer, you should probably take this line out.
    # if use_tpu:
    new_global_step = global_step + 1
    train_op = tf.group(train_op, [global_step.assign(new_global_step)])
    return train_op

from tensorflow.python.training import training_ops
class AdamWeightDecayOptimizer4(tf.train.Optimizer):
    """An implementation of the AdamWeightDecayOptimizer using tensorflow's standard Optimizer API.
    This one uses the C++ TF ops for better performance
    Inside the C++ Adam op implementation, there is:
    $$lr_t := \text{learning\_rate} * \sqrt{1 - beta_2^t} / (1 - beta_1^t)$$
    while the original BERT implementation doesn't have this. Therefore, it will be slightly different"""

    def __init__(self,
                 learning_rate=0.001,
                 weight_decay_rate=0.0,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-6,
                 exclude_from_weight_decay=None,
                 use_locking=False,
                 name='AdamWeightDecayOptimizer4'):
        super(AdamWeightDecayOptimizer4, self).__init__(use_locking, name)

        self._learning_rate = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon
        self._weight_decay_rate = weight_decay_rate
        self.exclude_from_weight_decay = exclude_from_weight_decay

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if not self._weight_decay_rate:
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

    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        for v in var_list:
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)

    #
    # In the base class, distributed_apply will try to call each variable's processor.
    # And depends on the type of the variable, the dense/sparse/resource versions of apply will be executed.
    # ResourceVariable is the new Variable implementation, which will replace the old one in TF 2.0
    #

    def _apply_dense(self, grad, var):
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")
        decayed_var = var
        if self._do_use_weight_decay(self._get_variable_name(var.name)):
            decayed_var = self._weight_decay_rate * var
        return training_ops.apply_adam(
            decayed_var, m, v,
            tf.cast(self._beta1, var.dtype.base_dtype),
            tf.cast(self._beta2, var.dtype.base_dtype),
            tf.cast(self._learning_rate, var.dtype.base_dtype),
            tf.cast(self._beta1, var.dtype.base_dtype),
            tf.cast(self._beta2, var.dtype.base_dtype),
            tf.cast(self._epsilon, var.dtype.base_dtype),
            grad, use_locking=self._use_locking).op

    def _resource_apply_dense(self, grad, var):
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")
        decayed_var = var
        if self._do_use_weight_decay(self._get_variable_name(var.name)):
            decayed_var = self._weight_decay_rate * var
        return training_ops.resource_apply_adam(
            decayed_var, m, v,
            tf.cast(self._beta1, var.dtype.base_dtype),
            tf.cast(self._beta2, var.dtype.base_dtype),
            tf.cast(self._learning_rate, var.dtype.base_dtype),
            tf.cast(self._beta1, var.dtype.base_dtype),
            tf.cast(self._beta2, var.dtype.base_dtype),
            tf.cast(self._epsilon, var.dtype.base_dtype),
            grad, use_locking=self._use_locking).op

    def _apply_sparse_shared(self, grad, var, indices, scatter_add):
        m = self.get_slot(var, "m")
        m_t = tf.assign(m, m * self._beta1, use_locking=self._use_locking)
        m_t = scatter_add(m, indices, grad * (1 - self._beta1))

        v = self.get_slot(var, "v")
        v_t = tf.assign(v, v * self._beta2, use_locking=self._use_locking)
        v_t = scatter_add(v, indices, (grad * grad) * (1 - self._beta2))

        update = m_t / (tf.sqrt(v_t) + self._epsilon)
        if self._do_use_weight_decay(self._get_variable_name(var.name)):
            update += self._weight_decay_rate * var
        update_with_lr = self._learning_rate * update

        var_update = tf.assign_sub(var, update_with_lr, use_locking=self._use_locking)

        return tf.group(*[var_update, m_t, v_t])

    def _apply_sparse(self, grad, var):
        return self._apply_sparse_shared(
            grad.values, var, grad.indices,
            lambda x, i, v: state_ops.scatter_add(  # pylint: disable=g-long-lambda
                x, i, v, use_locking=self._use_locking))

    def _resource_scatter_add(self, x, i, v):
        #
        # We use x.handle for ResourceVariables.
        # resource_scatter_add and scatter_add refer to different ops in C++.
        #
        with tf.control_dependencies(
                [resource_variable_ops.resource_scatter_add(x.handle, i, v)]):
            return x.value()

    def _resource_apply_sparse(self, grad, var, indices):
        return self._apply_sparse_shared(grad, var, indices,
                                         self._resource_scatter_add)



class AdamWeightDecayOptimizer3(tf.train.Optimizer):
    """An implementation of the AdamWeightDecayOptimizer using tensorflow's standard Optimizer API."""
    def __init__(self,
                 learning_rate=0.001,
                 weight_decay_rate=0.0,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-6,
                 exclude_from_weight_decay=None,
                 use_locking=False,
                 name='AdamWeightDecayOptimizer3'):
        super(AdamWeightDecayOptimizer3, self).__init__(use_locking, name)

        self._learning_rate = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon
        self._weight_decay_rate = weight_decay_rate
        self.exclude_from_weight_decay = exclude_from_weight_decay

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if not self._weight_decay_rate:
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

    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        for v in var_list:
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)

    #
    # In the base class, distributed_apply will try to call each variable's processor.
    # And depends on the type of the variable, the dense/sparse/resource versions of apply will be executed.
    # ResourceVariable is the new Variable implementation, which will replace the old one in TF 2.0
    #

    def _apply_dense(self, grad, var):
        #
        # Code here gets lots of OOM. Recommended GPU memory >= 12GB.
        # Mix of 8GB and 12GB GPUs will likely give OOM too. To avoid this problem, pass a smaller batch size in command line.
        #
        m = self.get_slot(var, "m")
        m_t = tf.multiply(m, self._beta1) + tf.multiply(grad, 1 - self._beta1)

        v = self.get_slot(var, "v")
        v_t = tf.multiply(v, self._beta2) + tf.multiply(tf.square(grad), 1 - self._beta2)

        update = m_t / (tf.sqrt(v_t) + self._epsilon)
        if self._do_use_weight_decay(self._get_variable_name(var.name)):
            update += self._weight_decay_rate * var
        update_with_lr = self._learning_rate * update

        var_update = tf.assign_sub(var, update_with_lr, use_locking=self._use_locking)

        return tf.group(*[var_update, tf.assign(m, m_t), tf.assign(v, v_t)])

    def _resource_apply_dense(self, grad, var):
        # Resource variable has similar API to old TF Variable.
        return self._apply_dense(grad, var)

    def _apply_sparse_shared(self, grad, var, indices, scatter_add):
        m = self.get_slot(var, "m")
        m_t = tf.assign(m, m * self._beta1, use_locking=self._use_locking)
        m_t = scatter_add(m, indices, grad * (1 - self._beta1))

        v = self.get_slot(var, "v")
        v_t = tf.assign(v, v * self._beta2, use_locking=self._use_locking)
        v_t = scatter_add(v, indices, (grad * grad) * (1 - self._beta2))

        update = m_t / (tf.sqrt(v_t) + self._epsilon)
        if self._do_use_weight_decay(self._get_variable_name(var.name)):
            update += self._weight_decay_rate * var
        update_with_lr = self._learning_rate * update

        var_update = tf.assign_sub(var, update_with_lr, use_locking=self._use_locking)

        return tf.group(*[var_update, m_t, v_t])

    def _apply_sparse(self, grad, var):
        return self._apply_sparse_shared(
            grad.values, var, grad.indices,
            lambda x, i, v: state_ops.scatter_add(  # pylint: disable=g-long-lambda
                x, i, v, use_locking=self._use_locking))

    def _resource_scatter_add(self, x, i, v):
        #
        # We use x.handle for ResourceVariables.
        # resource_scatter_add and scatter_add refer to different ops in C++.
        #
        with tf.control_dependencies(
                [resource_variable_ops.resource_scatter_add(x.handle, i, v)]):
            return x.value()

    def _resource_apply_sparse(self, grad, var, indices):
        return self._apply_sparse_shared(grad, var, indices,
                                         self._resource_scatter_add)


class AdamWeightDecayOptimizer2(optimizer_v2.OptimizerV2):
    """An implementation of the AdamWeightDecayOptimizer using tensorflow's newer OptimizerV2 API."""
    """However, there seems to be some incompatibility issues between OptimizerV2 and tf.distribute, which reports a duplicated node name error in runtime."""
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

    #
    # Optimizer's distributed_apply will try to call each variable's processor
    # and based on the type of variable, dense/sparse apply will be called.
    #

    def _apply_dense(self, grad, var, state):
        #
        # Here we simply replace scatter_add with addition
        #
        lr = state.get_hyper("learning_rate", var.dtype.base_dtype)
        beta1 = state.get_hyper("beta1", var.dtype.base_dtype)
        beta2 = state.get_hyper("beta2", var.dtype.base_dtype)
        epsilon = state.get_hyper("epsilon", var.dtype.base_dtype)
        weight_decay_rate = state.get_hyper('weight_decay_rate', var.dtype.base_dtype)
        #
        # Code here gets lots of OOM. Recommended GPU memory >= 12GB.
        # Mix of 8GB and 12GB GPUs will likely give OOM too. To avoid this problem, pass a smaller batch size in command line.
        #
        m = self.get_slot(var, "m")
        m_t = tf.multiply(m, beta1) + tf.multiply(grad, 1 - beta1)

        v = self.get_slot(var, "v")
        v_t = tf.multiply(v, beta2) + tf.multiply(tf.square(grad), 1 - beta2)

        update = m_t / (tf.sqrt(v_t) + epsilon)
        if self._do_use_weight_decay(self._get_variable_name(var.name)):
            update += self._weight_decay_rate * var
        update_with_lr = lr * update

        var_update = tf.assign_sub(var, update_with_lr, use_locking=self._use_locking)

        return tf.group(*[var_update, tf.assign(m, m_t), tf.assign(v, v_t)])

    def _resource_apply_dense(self, grad, var, state):
        # dense tensor of ResoureVariable should offer same API as Variable
        return self._apply_dense(grad, var, state)

    def _apply_sparse_shared(self, grad, var, indices, scatter_add, state):
        lr = state.get_hyper("learning_rate", var.dtype.base_dtype)
        beta1 = state.get_hyper("beta1", var.dtype.base_dtype)
        beta2 = state.get_hyper("beta2", var.dtype.base_dtype)
        epsilon = state.get_hyper("epsilon", var.dtype.base_dtype)
        weight_decay_rate = state.get_hyper('weight_decay_rate', var.dtype.base_dtype)
        m = self.get_slot(var, "m")
        m_t = tf.assign(m, m * beta1, use_locking=self._use_locking)
        m_t = scatter_add(m, indices, grad * (1 - beta1))

        v = self.get_slot(var, "v")
        v_t = tf.assign(v, v * beta2, use_locking=self._use_locking)
        v_t = scatter_add(v, indices, (grad * grad) * (1 - beta2))

        update = m_t / (tf.sqrt(v_t) + epsilon)
        if self._do_use_weight_decay(self._get_variable_name(var.name)):
            update += self._weight_decay_rate * var
        update_with_lr = lr * update

        var_update = tf.assign_sub(var, update_with_lr, use_locking=self._use_locking)

        return tf.group(*[var_update, m_t, v_t])

    def _apply_sparse(self, grad, var, state):
        return self._apply_sparse_shared(
            grad.values, var, grad.indices,
            lambda x, i, v: state_ops.scatter_add(  # pylint: disable=g-long-lambda
                x, i, v, use_locking=self._use_locking),
            state)

    def _resource_scatter_add(self, x, i, v):
        #
        # Daqi - handles incompatibility between the old Variable and new ResourceVariable. For now they refer to different C++ implementations.
        # Future releases should see a merge of scatter_add and resource_scatter_add.
        #
        with tf.control_dependencies(
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
            assignments += [param.assign(next_param),
                            m.assign(next_m), v.assign(next_v)]
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
