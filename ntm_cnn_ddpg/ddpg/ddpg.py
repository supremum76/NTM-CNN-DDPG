"""
Deep Deterministic Policy Gradient (DDPG).

Based on the implementation of the keras development team
https://keras.io/examples/rl/ddpg_pendulum/#introduction
https://github.com/keras-team/keras-io/blob/master/examples/rl/ddpg_pendulum.py
https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/rl/ipynb/ddpg_pendulum.ipynb
"""
from typing import Callable

import numpy as np
import tensorflow as tf

from ntm_cnn_ddpg.cnn.model import Model, Tensor


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

        self.state_buffer: list[Tensor | None] = [None for _ in range(self.buffer_capacity)]
        self.action_buffer: list[Tensor | None] = [None for _ in range(self.buffer_capacity)]
        self.reward_buffer: list[Tensor | None] = [None for _ in range(self.buffer_capacity)]
        self.next_state_buffer: list[Tensor | None] = [None for _ in range(self.buffer_capacity)]

    def record(self, state: Tensor, action: Tensor, reward: float, next_state: Tensor) -> None:
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

        self.state_buffer[index] = tf.cast(state, tf.float32)
        self.action_buffer[index] = tf.cast(action, tf.float32)
        self.reward_buffer[index] = tf.constant(reward, tf.float32)
        self.next_state_buffer[index] = tf.cast(next_state, tf.float32)

        self.buffer_counter += 1
        if self.buffer_counter == self.buffer_capacity * 2:
            self.buffer_counter = self.buffer_capacity

    def new_batch_records(self, batch_size: int, only_state_batch: bool) \
            -> tuple[Tensor, Tensor, Tensor, Tensor] | Tensor:
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
        state_batch: Tensor = tf.concat(
            values=[tf.reshape(self.state_buffer[i], [1, *self.state_buffer[i].shape])
                    for i in batch_indices],
            axis=0)

        if only_state_batch:
            return state_batch
        else:
            action_batch: Tensor = tf.concat(
                values=[tf.reshape(self.action_buffer[i], [1, *self.action_buffer[i].shape])
                        for i in batch_indices],
                axis=0)

            reward_batch: Tensor = tf.concat(values=[tf.reshape(self.reward_buffer[i], [1, 1])
                                                     for i in batch_indices],
                                             axis=0)
            next_state_batch: Tensor = tf.concat(values=[tf.reshape(self.next_state_buffer[i],
                                                                    [1, *self.next_state_buffer[i].shape])
                                                         for i in batch_indices],
                                                 axis=0)

            return state_batch, action_batch, reward_batch, next_state_batch


class DDPG:
    def __init__(self,
                 target_critic: Model,
                 target_actor: Model,
                 critic_model: Model,
                 actor_model: Model,
                 critic_model_input_concat: Callable[[Tensor, Tensor], Tensor]):
        """
        Инициализация
        :param target_critic: Целевая модель критика
        :param target_actor: Целевая модель актора
        :param critic_model: Модель критика
        :param actor_model: Модель актора
        :param critic_model_input_concat: Функция, объединяющая тензор состояния и тензор действия в единый тензор
        входа для модели критика. Интерфейс функции (state batch, action batch) -> critic model input batch.
        """
        self.target_critic = target_critic
        self.target_actor = target_actor
        self.critic_model = critic_model
        self.actor_model = actor_model
        self.critic_model_input_concat = critic_model_input_concat

    def policy(self, state: Tensor) -> Tensor:
        """
         Returns an action sampled from our Actor network
        :param state: State
        :return: Action
        """
        return self.actor_model.predict(model_input=state, training=False)

    def learn(self, buffer: Buffer, batch_size: int, epochs: int,
              q_learning_discount_factor: float,
              target_model_update_rate: float,
              critic_optimizer: tf.optimizers.Optimizer,
              actor_optimizer: tf.optimizers.Optimizer):
        for _ in range(epochs):
            self._update_critic_model(*buffer.new_batch_records(batch_size, only_state_batch=False),
                                      critic_optimizer=critic_optimizer,
                                      q_learning_discount_factor=tf.constant(q_learning_discount_factor))

        for _ in range(epochs):
            self._update_actor_model(state_batch=buffer.new_batch_records(batch_size, only_state_batch=True),
                                     actor_optimizer=actor_optimizer)

        tensor_update_rate: Tensor = tf.constant(target_model_update_rate)
        self._update_target(self.target_actor.trainable_variables, self.actor_model.trainable_variables,
                            tensor_update_rate)
        self._update_target(self.target_critic.trainable_variables, self.critic_model.trainable_variables,
                            tensor_update_rate)

    # This update target parameters slowly
    # Based on rate target_model_update_rate, which is much less than one.
    @tf.function
    def _update_target(self, target_weights: Tensor, weights: Tensor, update_rate: Tensor):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * update_rate + a * (1.0 - update_rate))

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def _update_critic_model(
            self, state_batch: Tensor, action_batch: Tensor, reward_batch: Tensor, next_state_batch: Tensor,
            critic_optimizer: tf.optimizers.Optimizer,
            q_learning_discount_factor: Tensor
    ):
        # Training and updating Critic network.
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            for t in self.critic_model.trainable_variables:
                tape.watch(t)

            target_actions = self.target_actor.predict(next_state_batch, training=False)
            y = reward_batch + q_learning_discount_factor * self.target_critic.predict(
                self.critic_model_input_concat(next_state_batch, target_actions), training=False
            )
            critic_value = self.critic_model.predict(self.critic_model_input_concat(state_batch, action_batch),
                                                     training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self.critic_model.trainable_variables)

        critic_optimizer.apply_gradients(
            zip(critic_grad, self.critic_model.trainable_variables)
        )

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def _update_actor_model(self, state_batch: Tensor, actor_optimizer: tf.optimizers.Optimizer):
        # Training and updating Actor network.
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            for t in self.actor_model.trainable_variables:
                tape.watch(t)

            actions = self.actor_model.predict(state_batch, training=True)
            critic_value = self.critic_model.predict(self.critic_model_input_concat(state_batch, actions),
                                                     training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor_model.trainable_variables)
        )
