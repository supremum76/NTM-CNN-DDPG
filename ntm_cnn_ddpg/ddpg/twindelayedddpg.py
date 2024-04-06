"""
Deep Deterministic Policy Gradient (DDPG).

Based on the implementation of the keras development team
https://keras.io/examples/rl/ddpg_pendulum/#introduction
https://github.com/keras-team/keras-io/blob/master/examples/rl/ddpg_pendulum.py
https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/rl/ipynb/ddpg_pendulum.ipynb
"""
import collections.abc
from dataclasses import dataclass
from typing import Callable, Optional, Any, List
from copy import deepcopy

import numpy as np
import tensorflow as tf
from scipy.optimize import minimize_scalar

from ntm_cnn_ddpg.cnn.model import Tensor, OptionalSeqTensors, SeqTensors, CriticModel, ActorModel


def _tensors_to_seq_tensors(tensors: OptionalSeqTensors) -> SeqTensors:
    if isinstance(tensors, collections.abc.Sequence):
        return tensors
    else:
        return [tensors]


class OUActionNoise:
    """
    To implement better exploration by the Actor network, we use noisy perturbations, specifically an
    Ornstein-Uhlenbeck process for generating noise, as described in the paper.
    It samples noise from a correlated normal distribution.
    """

    def __init__(self, mean: Tensor, std_deviation: Tensor, theta=0.15, dt=1e-2, x_initial: Tensor | None = None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self) -> Tensor:
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
                self.x_prev
                + self.theta * (self.mean - self.x_prev) * self.dt
                + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


class Buffer:
    def __init__(self, buffer_capacity: int):
        """
        Инициализация
        :param buffer_capacity: Максимальное количество хранимых наблюдений
        """
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        self.state_buffer: list[OptionalSeqTensors | None] = [None for _ in range(self.buffer_capacity)]
        self.action_buffer: list[OptionalSeqTensors | None] = [None for _ in range(self.buffer_capacity)]
        self.reward_buffer: list[OptionalSeqTensors | None] = [None for _ in range(self.buffer_capacity)]
        self.next_state_buffer: list[OptionalSeqTensors | None] = [None for _ in range(self.buffer_capacity)]

    def record(self, state: OptionalSeqTensors, action: OptionalSeqTensors, reward: float,
               next_state: OptionalSeqTensors) -> None:
        """
        Takes and remember state, action, reward and next state
        :param state: state environment
        :param action: action
        :param reward: reward scalar quantity
        :param next_state: next state environment
        :return: None
        """
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index: int = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = tuple(
            map(lambda tensor: tf.cast(tensor, tf.float32), _tensors_to_seq_tensors(state)))
        self.action_buffer[index] = tuple(
            map(lambda tensor: tf.cast(tensor, tf.float32), _tensors_to_seq_tensors(action)))
        self.reward_buffer[index] = tf.constant(reward, tf.float32)
        self.next_state_buffer[index] = tuple(
            map(lambda tensor: tf.cast(tensor, tf.float32), _tensors_to_seq_tensors(next_state)))

        self.buffer_counter += 1
        if self.buffer_counter == self.buffer_capacity * 2:
            self.buffer_counter = self.buffer_capacity

    def new_batch_records(self, batch_size: int, only_state_batch: bool) \
            -> tuple[OptionalSeqTensors, OptionalSeqTensors, Tensor, OptionalSeqTensors] | OptionalSeqTensors:
        """
        Return a new randomized batch of records for models learn
        :param batch_size: batch size
        :param only_state_batch: If only_state_batch then returned only batch for state.
        :return: batch of records in format (state, action, reward, next state) if not only_state_batch.
        If only_state_batch then returned only batch for state.
        """
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)

        # Randomly sample indices
        batch_indices = np.random.choice(record_range, min(record_range, batch_size))

        # collect the batch
        state_tensors_count: int = len(self.state_buffer[0])
        state_batch: OptionalSeqTensors = []
        for k in range(state_tensors_count):
            state_batch.append(tf.concat(
                values=[tf.reshape(self.state_buffer[i][k], [1, *self.state_buffer[i][k].shape])
                        for i in batch_indices],
                axis=0))

        if only_state_batch:
            return state_batch
        else:
            action_tensors_count: int = len(self.action_buffer[0])
            action_batch: OptionalSeqTensors = []
            for k in range(action_tensors_count):
                action_batch.append(tf.concat(
                    values=[tf.reshape(self.action_buffer[i][k], [1, *self.action_buffer[i][k].shape])
                            for i in batch_indices],
                    axis=0))

            reward_batch: Tensor = tf.concat(values=[tf.reshape(self.reward_buffer[i], [1, 1])
                                                     for i in batch_indices],
                                             axis=0)

            next_state_batch: OptionalSeqTensors = []
            for k in range(state_tensors_count):
                next_state_batch.append(tf.concat(
                    values=[tf.reshape(self.next_state_buffer[i][k], [1, *self.next_state_buffer[i][k].shape])
                            for i in batch_indices],
                    axis=0))

            return state_batch, action_batch, reward_batch, next_state_batch

    def records_count(self) -> int:
        return min(self.buffer_counter, self.buffer_capacity)


@dataclass(frozen=True)
class LearningRateSearchOptions:
    bounds_lower: float
    bounds_upper: float
    max_iterations: int
    tol: float


class TwinDelayedDDPG:
    """
    Реализация Twin Delayed DDPG (другое наименование TD3)
    """

    def __init__(self,
                 td3_target_critic1: CriticModel,
                 td3_target_critic2: CriticModel,
                 td3_critic_model1: CriticModel,
                 td3_critic_model2: CriticModel,
                 target_actor: ActorModel,
                 actor_model: ActorModel,
                 critic_model_input_concat: Callable[[OptionalSeqTensors, OptionalSeqTensors],
                 OptionalSeqTensors]):
        """
        Инициализация
        :param td3_target_critic1: Целевая модель критика #1 для реализации Twin Delayed DDPG (TD3)
        :param td3_target_critic2: Целевая модель критика #2 для реализации Twin Delayed DDPG (TD3)
        :param td3_critic_model1: Модель критика #1 для реализации Twin Delayed DDPG (TD3)
        :param td3_critic_model2: Модель критика #2 для реализации Twin Delayed DDPG (TD3)
        :param target_actor: Целевая модель актора
        :param actor_model: Модель актора
        :param critic_model_input_concat: Функция, объединяющая тензоры состояния и тензоры действия во
        вход для модели критика. Интерфейс функции (state batch, action batch) -> critic model input batch.
        """
        self.td3_target_critic1 = td3_target_critic1
        self.td3_target_critic2 = td3_target_critic2
        self.td3_critic_model1 = td3_critic_model1
        self.td3_critic_model2 = td3_critic_model2
        self.target_actor = target_actor
        self.actor_model = actor_model
        self.critic_model_input_concat = critic_model_input_concat
        self._learn_iter = 0

    def learn(self, buffer: Buffer, batch_size: int, epochs: int,
              q_learning_discount_factor: float,
              target_model_update_rate: float,
              td3_critic_optimizer1: tf.optimizers.Optimizer,
              td3_critic_optimizer2: tf.optimizers.Optimizer,
              actor_optimizer: Optional[tf.optimizers.Optimizer],
              td3_target_policy_smoothing: Optional[Callable[[OptionalSeqTensors], OptionalSeqTensors]] = None,
              td3_policy_update_delay: int = 1,
              learning_rate_search_options: Optional[LearningRateSearchOptions] = None
              ):

        # if not enough records for batch
        if buffer.records_count() < batch_size:
            return

        self._learn_iter += 1

        for _ in range(epochs):
            self._update_critic_model(*buffer.new_batch_records(batch_size, only_state_batch=False),
                                      td3_critic_optimizer1=td3_critic_optimizer1,
                                      td3_critic_optimizer2=td3_critic_optimizer2,
                                      q_learning_discount_factor=tf.constant(q_learning_discount_factor),
                                      td3_target_policy_smoothing=td3_target_policy_smoothing,
                                      learning_rate_search_options=learning_rate_search_options)

        if self._learn_iter % td3_policy_update_delay == 0:
            for _ in range(epochs):
                self._update_actor_model(state_batch=buffer.new_batch_records(batch_size, only_state_batch=True),
                                         actor_optimizer=actor_optimizer,
                                         learning_rate_search_options=learning_rate_search_options)

        tensor_update_rate: Tensor = tf.constant(target_model_update_rate)

        self._update_target(self.target_actor.trainable_variables, self.actor_model.trainable_variables,
                            tensor_update_rate)
        self._update_target(self.td3_target_critic1.trainable_variables, self.td3_critic_model1.trainable_variables,
                            tensor_update_rate)
        self._update_target(self.td3_target_critic2.trainable_variables, self.td3_critic_model2.trainable_variables,
                            tensor_update_rate)

    def policy(self, state: OptionalSeqTensors) -> OptionalSeqTensors:
        """
         Returns an action sampled from our Actor network
        :param state: State
        :return: Action
        """
        return self.actor_model.predict(model_input=state, training=False, batch_mode=False)

    # This update target parameters slowly
    # Based on rate target_model_update_rate, which is much less than one.
    @tf.function
    def _update_target(self, target_weights: Tensor, weights: Tensor, update_rate: Tensor):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * update_rate + a * (1.0 - update_rate))

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    def _update_critic_model(
            self, state_batch: OptionalSeqTensors, action_batch: OptionalSeqTensors, reward_batch: Tensor,
            next_state_batch: OptionalSeqTensors,
            td3_critic_optimizer1: tf.optimizers.Optimizer,
            td3_critic_optimizer2: tf.optimizers.Optimizer,
            q_learning_discount_factor: Tensor,
            td3_target_policy_smoothing: Callable[[OptionalSeqTensors], OptionalSeqTensors],
            learning_rate_search_options: Optional[LearningRateSearchOptions] = None
    ):

        # Training and updating Critic network.
        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
            for t in self.td3_critic_model1.trainable_variables + self.td3_critic_model2.trainable_variables:
                tape.watch(t)

            target_actions = self.target_actor.predict(next_state_batch, training=False)
            if td3_target_policy_smoothing:
                target_actions = td3_target_policy_smoothing(target_actions)

            critic_input = self.critic_model_input_concat(next_state_batch, target_actions)
            td3_next_reward = tf.math.minimum(
                self.td3_target_critic1.predict(model_input=critic_input, training=False),
                self.td3_target_critic2.predict(model_input=critic_input, training=False)
            )

            y = reward_batch + q_learning_discount_factor * td3_next_reward

            critic_input = self.critic_model_input_concat(state_batch, action_batch)
            critic_value1 = self.td3_critic_model1.predict(model_input=critic_input, training=True)
            critic_value2 = self.td3_critic_model2.predict(model_input=critic_input, training=True)
            critic_loss1 = tf.math.reduce_mean(tf.math.square(y - critic_value1))
            critic_loss2 = tf.math.reduce_mean(tf.math.square(y - critic_value2))

        grads_and_vars1 = td3_critic_optimizer1.compute_gradients(loss=critic_loss1,
                                                                  var_list=self.td3_critic_model1.trainable_variables,
                                                                  tape=tape)
        grads_and_vars2 = td3_critic_optimizer2.compute_gradients(loss=critic_loss2,
                                                                  var_list=self.td3_critic_model2.trainable_variables,
                                                                  tape=tape)

        def optimal_learning_rate(critic_model, grads_and_vars, critic_optimizer) -> float:
            def fun(x: float) -> float:
                # _distribution_strategy не может клонироваться
                strategy = critic_optimizer._distribution_strategy
                critic_optimizer._distribution_strategy = None  # TODO сделать проверку наличия атрибута _distribution_strategy
                optimizer = deepcopy(critic_optimizer)
                optimizer._distribution_strategy = strategy
                critic_optimizer._distribution_strategy = strategy

                # сохраняем текущие параметры модели
                model_weights = critic_model.get_weights()
                optimizer.learning_rate = x
                optimizer.apply_gradients(grads_and_vars)
                critic_value = critic_model.predict(self.critic_model_input_concat(state_batch, action_batch),
                                                    training=False)
                critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))
                # восстанавливаем параметры модели
                critic_model.set_weights(model_weights)
                return critic_loss.numpy()

            r = minimize_scalar(fun=fun,
                                bounds=[learning_rate_search_options.bounds_lower,
                                        learning_rate_search_options.bounds_upper],
                                method="bounded",
                                options={
                                    "maxiter": learning_rate_search_options.max_iterations,
                                    "xatol": learning_rate_search_options.tol
                                })

            return r.x

        if learning_rate_search_options:
            td3_critic_optimizer1.learning_rate = optimal_learning_rate(
                critic_model=self.td3_critic_model1,
                grads_and_vars=grads_and_vars1,
                critic_optimizer=td3_critic_optimizer1)

            td3_critic_optimizer2.learning_rate = optimal_learning_rate(
                critic_model=self.td3_critic_model2, grads_and_vars=grads_and_vars2,
                critic_optimizer=td3_critic_optimizer2)

        td3_critic_optimizer1.apply_gradients(grads_and_vars1)
        td3_critic_optimizer2.apply_gradients(grads_and_vars2)

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    def _update_actor_model(self,
                            state_batch: OptionalSeqTensors,
                            actor_optimizer: tf.optimizers.Optimizer,
                            learning_rate_search_options: Optional[LearningRateSearchOptions] = None):

        """
        target_actions = _tensors_to_seq_tensors(optimal_action(self.critic_model, state_batch))
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            for t in self.actor_model.trainable_variables:
                tape.watch(t)
            actor_actions = _tensors_to_seq_tensors(self.actor_model.predict(state_batch, training=True))
            actor_loss = tf.math.reduce_mean(
                tf.math.square(tf.stack(target_actions, axis=0) - tf.stack(actor_actions, axis=0)))
        """

        # Training and updating Actor network.
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            for t in self.actor_model.trainable_variables:
                tape.watch(t)

            actions = self.actor_model.predict(state_batch, training=True)
            critic_value = self.td3_critic_model1.predict(self.critic_model_input_concat(state_batch, actions),
                                                          training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        grads_and_vars = actor_optimizer.compute_gradients(loss=actor_loss,
                                                           var_list=self.actor_model.trainable_variables,
                                                           tape=tape)

        def optimal_learning_rate() -> float:
            def fun(x: float) -> float:
                # _distribution_strategy не может клонироваться
                strategy = actor_optimizer._distribution_strategy
                actor_optimizer._distribution_strategy = None
                optimizer = deepcopy(actor_optimizer)  # actor_optimizer.from_config(actor_optimizer.get_config())
                optimizer._distribution_strategy = strategy
                actor_optimizer._distribution_strategy = strategy

                # сохраняем текущие параметры модели
                model_weights = self.actor_model.get_weights()
                optimizer.learning_rate = x
                optimizer.apply_gradients(grads_and_vars)
                actions = self.actor_model.predict(state_batch, training=False)
                critic_value = self.td3_critic_model1.predict(self.critic_model_input_concat(state_batch, actions),
                                                              training=False)
                # Used `-value` as we want to maximize the value given
                # by the critic for our actions
                actor_loss = -tf.math.reduce_mean(critic_value)
                # восстанавливаем параметры модели
                self.actor_model.set_weights(model_weights)
                return actor_loss.numpy()

            r = minimize_scalar(fun=fun,
                                bounds=[learning_rate_search_options.bounds_lower,
                                        learning_rate_search_options.bounds_upper],
                                method="bounded",
                                options={
                                    "maxiter": learning_rate_search_options.max_iterations,
                                    "xatol": learning_rate_search_options.tol
                                })
            # r = minimize_scalar(fun=fun, bracket=[1E-4, 1E-2], method="brent", tol=1E-3, options={"maxiter": 30})

            return r.x

        if learning_rate_search_options:
            actor_optimizer.learning_rate = optimal_learning_rate()
        actor_optimizer.apply_gradients(grads_and_vars)
        # actor_optimizer.minimize(loss=actor_loss, var_list=self.actor_model.trainable_variables, tape=tape)
