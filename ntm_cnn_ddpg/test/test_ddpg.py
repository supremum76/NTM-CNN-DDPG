import itertools
import random
import time
from collections import namedtuple
from copy import deepcopy
from enum import Enum
from unittest import TestCase

import tensorflow as tf
from keras import layers
from keras.optimizers import Optimizer

from ntm_cnn_ddpg.cnn.model import Tensor, Model
from ntm_cnn_ddpg.cnn.model2d import Model2D, concat_input_tensors
from ntm_cnn_ddpg.ddpg.ddpg import Buffer, DDPG, OUActionNoise
from ntm_cnn_ddpg.ntm.controller.memory_bank import MemoryBank


class TicTacToeGameStatus(Enum):
    DRAW = 0
    CROSS_WON = 1
    ZERO_WON = 2
    THE_GAME_CONTINUES = -1


class TicTacToePosStatus(Enum):
    EMPTY = 0
    CROSS = 1
    ZERO = -1


Point2D = namedtuple('Point2D', ['r', 'c'])


class DenseModel(Model):
    def __init__(self, tf_model, output_length):
        self.__model = tf_model
        self.output_length = output_length

    def predict(self, model_input: Tensor, training: bool) -> Tensor:
        batch_mode: bool = model_input.shape.ndims == 2

        result: Tensor = self.__model(inputs=
                                      model_input if batch_mode else
                                      # добавляем фиктивное измерение пакета
                                      tf.reshape(tensor=model_input, shape=(1, *model_input.shape)),
                                      training=training)
        if not batch_mode:
            # удаляем фиктивное измерение пакета
            result = tf.reshape(tensor=result, shape=self.output_length)

        return result

    @property
    def trainable_variables(self) -> Tensor:
        return self.__model.trainable_variables


class TestDDPG(TestCase):

    def _test_tic_tac_toe_without_ntm(self) -> None:
        """
        Тест связки DDPG+CNN на способность обучится игре крестики нолики на поле 3x3.
        NTM не используется.
        Проверяется способность DDPG+CNN определить оптимальные правила поведения в игре.
        https://en.wikipedia.org/wiki/Tic-tac-toe
        :return:
        """

        def check_of_end_game(game_state: list[list[TicTacToePosStatus]]) -> TicTacToeGameStatus:
            transpose_game_state: list[list[TicTacToePosStatus]] = [list(x) for x in zip(*game_state)]

            if [TicTacToePosStatus.CROSS] * 3 in game_state \
                    or [TicTacToePosStatus.CROSS] * 3 in transpose_game_state \
                    or (game_state[0][0] == TicTacToePosStatus.CROSS
                        and game_state[1][1] == TicTacToePosStatus.CROSS
                        and game_state[2][2] == TicTacToePosStatus.CROSS) \
                    or (transpose_game_state[0][0] == TicTacToePosStatus.CROSS
                        and transpose_game_state[1][1] == TicTacToePosStatus.CROSS
                        and transpose_game_state[2][2] == TicTacToePosStatus.CROSS):
                return TicTacToeGameStatus.CROSS_WON
            elif [TicTacToePosStatus.ZERO] * 3 in game_state \
                    or [TicTacToePosStatus.ZERO] * 3 in transpose_game_state \
                    or (game_state[0][0] == TicTacToePosStatus.ZERO
                        and game_state[1][1] == TicTacToePosStatus.ZERO
                        and game_state[2][2] == TicTacToePosStatus.ZERO) \
                    or (transpose_game_state[0][0] == TicTacToePosStatus.ZERO
                        and transpose_game_state[1][1] == TicTacToePosStatus.ZERO
                        and transpose_game_state[2][2] == TicTacToePosStatus.ZERO):
                return TicTacToeGameStatus.ZERO_WON
            elif TicTacToePosStatus.EMPTY in itertools.chain(*game_state):
                return TicTacToeGameStatus.THE_GAME_CONTINUES
            else:
                return TicTacToeGameStatus.DRAW

        def opponent_action(game_state: list[list[TicTacToePosStatus]]) \
                -> Point2D | None:
            """
            Реализация стратегии опонента в виде случайного выбора позиции
            :param game_state: game state
            :return: Позиция хода опонента (row, column)
            """
            empty_pos: [tuple[int, int]] = [(r, c) for r in range(3) for c in range(3)
                                            if game_state[r][c] == TicTacToePosStatus.EMPTY]
            if empty_pos:
                return empty_pos[random.randrange(0, len(empty_pos))]
            else:
                return None

        def game_state_to_model_input(game_state: list[list[TicTacToePosStatus]]) -> Tensor:
            # Дополняем игровое поле до размера 4x4, чтобы сверточная 2D модель могла корректно обработать его.
            extended_game_state = deepcopy(game_state)
            extended_game_state[0].append(TicTacToePosStatus.EMPTY)
            extended_game_state[1].append(TicTacToePosStatus.EMPTY)
            extended_game_state[2].append(TicTacToePosStatus.EMPTY)
            extended_game_state.append([TicTacToePosStatus.EMPTY] * 4)

            return tf.reshape(
                tensor=tf.constant(list(map(lambda x: x.value, itertools.chain(*extended_game_state))),
                                   dtype=tf.float32),
                shape=(4, 4, 1)
            )

        # создаем модели критика и актора
        target_actor: Model2D = Model2D(
            input_2d_shape=(4, 4, 1),
            input_1d_shape=(0, 0),
            output_length=3 * 3,
            start_filters=10,
            start_kernel_size=(2, 2),
            start_pool_size=(2, 2),
            rate_of_filters_increase=1,
            rate_of_kernel_size_increase=0,
            rate_of_pool_size_increase=0
        )

        target_critic: Model2D = Model2D(
            input_2d_shape=(4, 4, 1),
            input_1d_shape=(3 * 3, 1),
            output_length=1,
            start_filters=10,
            start_kernel_size=(2, 2),
            start_pool_size=(2, 2),
            rate_of_filters_increase=1,
            rate_of_kernel_size_increase=0,
            rate_of_pool_size_increase=0
        )

        actor_model: Model2D = Model2D(
            input_2d_shape=(4, 4, 1),
            input_1d_shape=(0, 0),
            output_length=3 * 3,
            start_filters=10,
            start_kernel_size=(2, 2),
            start_pool_size=(2, 2),
            rate_of_filters_increase=1,
            rate_of_kernel_size_increase=0,
            rate_of_pool_size_increase=0
        )

        critic_model: Model2D = Model2D(
            input_2d_shape=(4, 4, 1),
            input_1d_shape=(3 * 3, 1),  # TODO неочевидна связь между выходом actor и входом critic. Необходима фабрика.
            output_length=1,
            start_filters=10,
            start_kernel_size=(2, 2),
            start_pool_size=(2, 2),
            rate_of_filters_increase=1,
            rate_of_kernel_size_increase=0,
            rate_of_pool_size_increase=0
        )

        # Learning rate for actor - critic models
        critic_optimizer: Optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)
        actor_optimizer: Optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        # содаем буфер обучающих примеров
        buffer: Buffer = Buffer(
            buffer_capacity=10000,
            batch_size=500,
            target_critic=target_critic,
            target_actor=target_actor,
            critic_model=critic_model,
            actor_model=actor_model,
            critic_optimizer=critic_optimizer,
            actor_optimizer=actor_optimizer,
            q_learning_discount_factor=0.9,
            critic_model_input_concat=lambda state_batch, action_batch:
            # добавляем к действиям фиктивное измерение фильтра,  чтобы привести их к форме, ожидаемой
            # функцией concat_input_tensors
            concat_input_tensors(state_batch, tf.reshape(action_batch, (*action_batch.shape, 1)))
        )

        ddpg: DDPG = DDPG(
            target_critic=target_critic,
            target_actor=target_actor,
            critic_model=critic_model,
            actor_model=actor_model,
            buffer=buffer,
            noise_object=OUActionNoise(mean=tf.constant(0.0), std_deviation=tf.constant(0.1)),
            target_model_update_rate=0.005
        )

        # Разыгрываем серию игр.
        # DDPG играет за кресты против условного игрока, который случайным образом заполняет нулями свободные ячейки.
        # Для каждой следующий 100 игр считаем % побед DDPG.
        # Состояним среды для DDPG является текущее состояние игрового поля.
        # Выход DDPG дополнительно обрабатывается функцией softmax.
        # Затем выбирается свободная позиция с максимальным значением вероятности.
        # В эту позицию DDPG помещает свой знак.
        win_rate: float = 0
        history_win_rate: list[float] = []
        game_state: list[list[TicTacToePosStatus]] = \
            [[TicTacToePosStatus.EMPTY] * 3, [TicTacToePosStatus.EMPTY] * 3, [TicTacToePosStatus.EMPTY] * 3]
        for game_num in range(100000):
            for r in range(3):
                for c in range(3):
                    game_state[r][c] = TicTacToePosStatus.EMPTY

            while True:
                original_action: Tensor = ddpg.policy(game_state_to_model_input(game_state))
                action: Tensor = tf.reshape(tensor=tf.keras.activations.softmax(
                    tf.reshape(tensor=original_action, shape=(1, 3 * 3))),
                    shape=(3, 3))

                # выбираем свободную позицию для хода
                point: Point2D | None = None
                max_prob: float = -1
                for r in range(3):
                    for c in range(3):
                        if game_state[r][c] == TicTacToePosStatus.EMPTY:
                            prob: float = float(action[r, c])
                            if prob > max_prob:
                                point = Point2D(r, c)
                                max_prob = prob

                # ход DDPG
                prev_game_state = deepcopy(game_state)
                if point:
                    game_state[point.r][point.c] = TicTacToePosStatus.CROSS
                else:
                    self.fail(msg="Game solution not selected")

                # проверяем статус игры
                game_status: TicTacToeGameStatus = check_of_end_game(game_state)
                reward: float = 0
                if game_status == TicTacToeGameStatus.CROSS_WON:
                    reward = 1

                if game_status == TicTacToeGameStatus.THE_GAME_CONTINUES:
                    r, c = opponent_action(game_state)
                    game_state[r][c] = TicTacToePosStatus.ZERO

                    # проверяем статус игры
                    game_status = check_of_end_game(game_state)

                    if game_status == TicTacToeGameStatus.ZERO_WON:
                        reward = -1

                ddpg.learn(
                    prev_state=game_state_to_model_input(prev_game_state),
                    action=original_action,
                    reward=reward,
                    next_state=game_state_to_model_input(game_state)
                )

                if game_status != TicTacToeGameStatus.THE_GAME_CONTINUES:
                    # собираем статистику по исходам игры
                    if game_status == TicTacToeGameStatus.CROSS_WON:
                        win_rate += 1

                    if game_num > 0 and game_num % 100 == 0:
                        history_win_rate.append(win_rate / 100)
                        win_rate = 0

                    break

        print(history_win_rate)

    def _test_tic_tac_toe_mlp_without_ntm(self) -> None:
        """
        Тест связки DDPG+MLP на способность обучится игре крестики нолики на поле 3x3.
        NTM не используется.
        Проверяется способность DDPG+MLP определить оптимальные правила поведения в игре.
        https://en.wikipedia.org/wiki/Tic-tac-toe
        :return:
        """

        def check_of_end_game(game_state: list[list[TicTacToePosStatus]]) -> TicTacToeGameStatus:
            transpose_game_state: list[list[TicTacToePosStatus]] = [list(x) for x in zip(*game_state)]

            if [TicTacToePosStatus.CROSS] * 3 in game_state \
                    or [TicTacToePosStatus.CROSS] * 3 in transpose_game_state \
                    or (game_state[0][0] == TicTacToePosStatus.CROSS
                        and game_state[1][1] == TicTacToePosStatus.CROSS
                        and game_state[2][2] == TicTacToePosStatus.CROSS) \
                    or (transpose_game_state[0][0] == TicTacToePosStatus.CROSS
                        and transpose_game_state[1][1] == TicTacToePosStatus.CROSS
                        and transpose_game_state[2][2] == TicTacToePosStatus.CROSS):
                return TicTacToeGameStatus.CROSS_WON
            elif [TicTacToePosStatus.ZERO] * 3 in game_state \
                    or [TicTacToePosStatus.ZERO] * 3 in transpose_game_state \
                    or (game_state[0][0] == TicTacToePosStatus.ZERO
                        and game_state[1][1] == TicTacToePosStatus.ZERO
                        and game_state[2][2] == TicTacToePosStatus.ZERO) \
                    or (transpose_game_state[0][0] == TicTacToePosStatus.ZERO
                        and transpose_game_state[1][1] == TicTacToePosStatus.ZERO
                        and transpose_game_state[2][2] == TicTacToePosStatus.ZERO):
                return TicTacToeGameStatus.ZERO_WON
            elif TicTacToePosStatus.EMPTY in itertools.chain(*game_state):
                return TicTacToeGameStatus.THE_GAME_CONTINUES
            else:
                return TicTacToeGameStatus.DRAW

        def opponent_action(game_state: list[list[TicTacToePosStatus]]) \
                -> Point2D | None:
            """
            Реализация стратегии опонента в виде случайного выбора позиции
            :param game_state: game state
            :return: Позиция хода опонента (row, column)
            """
            empty_pos: [tuple[int, int]] = [(r, c) for r in range(3) for c in range(3)
                                            if game_state[r][c] == TicTacToePosStatus.EMPTY]
            if empty_pos:
                return empty_pos[random.randrange(0, len(empty_pos))]
            else:
                return None

        def game_state_to_model_input(game_state: list[list[TicTacToePosStatus]]) -> Tensor:
            return tf.reshape(
                tensor=tf.constant(list(map(lambda x: x.value, itertools.chain(*game_state))),
                                   dtype=tf.float32),
                shape=(3 * 3)
            )

        def get_actor_dense():
            last_init = tf.random_uniform_initializer(minval=-1E-0, maxval=1E-0)

            # !!! не используем layers.BatchNormalization(), так как игра меняеется в ходе обучения Actor и
            # из-за этого смещается распределение входных сигналов. Это создает нестабильность восприятия игровой среды.

            inputs = layers.Input(shape=(3 * 3,))
            out = layers.Dense(10, activation="softsign", kernel_initializer=last_init)(inputs)
            out = layers.Dense(10, activation="softsign", kernel_initializer=last_init)(out)
            outputs = layers.Dense(9, activation="linear", kernel_initializer=last_init)(out)

            model = tf.keras.Model(inputs, outputs)
            return model

        def get_critic_dense():
            last_init = tf.random_uniform_initializer(minval=-1E-0, maxval=1E-0)

            # !!! не используем layers.BatchNormalization(), так как игра меняеется в ходе обучения Actor и
            # из-за этого смещается распределение входных сигналов. Это создает нестабильность восприятия игровой среды.

            inputs = layers.Input(shape=(3 * 3 + 9,))
            out = layers.Dense(10, activation="softsign", kernel_initializer=last_init)(inputs)
            out = layers.Dense(10, activation="softsign", kernel_initializer=last_init)(out)
            outputs = layers.Dense(1, activation="linear", kernel_initializer=last_init)(out)

            model = tf.keras.Model(inputs, outputs)

            return model

        target_actor = DenseModel(get_actor_dense(), 9)
        actor_model = DenseModel(get_actor_dense(), 9)
        target_critic = DenseModel(get_critic_dense(), 1)
        critic_model = DenseModel(get_critic_dense(), 1)

        # !!! weight_decay=1.0 Предотвращаем попадание входа функции активации в область "насыщения".
        critic_optimizer: Optimizer = tf.keras.optimizers.Adam(learning_rate=1E-3, weight_decay=1.0)
        actor_optimizer: Optimizer = tf.keras.optimizers.Adam(learning_rate=1E-3, weight_decay=1.0)

        # содаем буфер обучающих примеров
        buffer: Buffer = Buffer(buffer_capacity=50000)

        ddpg: DDPG = DDPG(
            target_critic=target_critic,
            target_actor=target_actor,
            critic_model=critic_model,
            actor_model=actor_model,
            critic_model_input_concat=lambda state_batch, action_batch:
            tf.concat(values=[state_batch, action_batch], axis=1)
        )

        exploration_noise: OUActionNoise = OUActionNoise(mean=tf.constant([0.0]*9), std_deviation=tf.constant([1.0]*9))

        """
        batch_size=10,
            target_critic=target_critic,
            target_actor=target_actor,
            critic_model=critic_model,
            actor_model=actor_model,
            critic_optimizer=critic_optimizer,
            actor_optimizer=actor_optimizer,
            q_learning_discount_factor=0.9,
            
            noise_object=OUActionNoise(mean=tf.constant([0.0]*9), std_deviation=tf.constant([0.1]*9)),
            
            target_model_update_rate=0.005
        """

        # Разыгрываем серию игр.
        # DDPG играет за кресты против условного игрока, который случайным образом заполняет нулями свободные ячейки.
        # Для каждой следующий 100 игр считаем % побед DDPG.
        # Состояним среды для DDPG является текущее состояние игрового поля.
        # Выход DDPG дополнительно обрабатывается функцией softmax.
        # Затем выбирается свободная позиция с максимальным значением вероятности.
        # В эту позицию DDPG помещает свой знак.
        win_rate: float = 0
        history_win_rate: list[float] = []
        game_state: list[list[TicTacToePosStatus]] = \
            [[TicTacToePosStatus.EMPTY] * 3, [TicTacToePosStatus.EMPTY] * 3, [TicTacToePosStatus.EMPTY] * 3]
        for game_num in range(100000):
            for r in range(3):
                for c in range(3):
                    game_state[r][c] = TicTacToePosStatus.EMPTY

            while True:
                original_action: Tensor = ddpg.policy(game_state_to_model_input(game_state))
                k: Tensor = tf.reduce_mean(tf.abs(original_action))
                noise: Tensor = exploration_noise() * (k * 0.2 if k > 0 else 1)
                original_action = original_action + noise

                action: Tensor = tf.reshape(tensor=tf.keras.activations.softmax(
                    tf.reshape(tensor=original_action, shape=(1, 3 * 3))),
                    shape=(3, 3))

                # выбираем свободную позицию для хода
                point: Point2D | None = None
                max_prob: float = -1
                for r in range(3):
                    for c in range(3):
                        if game_state[r][c] == TicTacToePosStatus.EMPTY:
                            prob: float = float(action[r, c])
                            if prob > max_prob:
                                point = Point2D(r, c)
                                max_prob = prob

                # ход DDPG
                prev_game_state = deepcopy(game_state)
                if point:
                    game_state[point.r][point.c] = TicTacToePosStatus.CROSS
                else:
                    self.fail(msg="Game solution not selected")

                # проверяем статус игры
                game_status: TicTacToeGameStatus = check_of_end_game(game_state)
                reward: float = 0
                if game_status == TicTacToeGameStatus.CROSS_WON:
                    reward = 1

                if game_status == TicTacToeGameStatus.THE_GAME_CONTINUES:
                    r, c = opponent_action(game_state)
                    game_state[r][c] = TicTacToePosStatus.ZERO

                    # проверяем статус игры
                    game_status = check_of_end_game(game_state)

                    if game_status == TicTacToeGameStatus.ZERO_WON:
                        reward = -1

                buffer.record(state=game_state_to_model_input(prev_game_state),
                              action=original_action,
                              reward=reward,
                              next_state=game_state_to_model_input(game_state))

                ddpg.learn(buffer=buffer,
                           batch_size=10,
                           epochs=5,
                           q_learning_discount_factor=0.9,
                           # target_model_update_rate=0.005,
                           target_model_update_rate=0.1,
                           critic_optimizer=critic_optimizer,
                           actor_optimizer=actor_optimizer
                           )

                if game_status != TicTacToeGameStatus.THE_GAME_CONTINUES:
                    # собираем статистику по исходам игры
                    if game_status == TicTacToeGameStatus.CROSS_WON:
                        win_rate += 1

                    if game_num > 0 and game_num % 100 == 0:
                        history_win_rate.append(win_rate / 100)
                        win_rate = 0
                        print(sum(history_win_rate)/len(history_win_rate))
                        print(sum(history_win_rate[len(history_win_rate) - 30:])/len(history_win_rate[len(history_win_rate) - 30:]))
                        print(history_win_rate[len(history_win_rate) - 10:])
                        print(len(history_win_rate))

                    break

        print(history_win_rate)

    def _test_tic_tac_toe_mlp_ntm(self) -> None:
        """
        Тест связки DDPG+MLP+NTM на способность обучится игре крестики нолики на поле 3x3.
        Проверяется способность DDPG+MLP+NTM определить оптимальные правила поведения в игре.
        https://en.wikipedia.org/wiki/Tic-tac-toe
        Цель эксперимента заключается в проверке способности ntm-контроллера обучится эффективному использованию
        внешнего банка памяти при использовании Deep Q-learning метода.

        DDPG по прежнему в явном виде получает текущийц статус игры.
        Однако NTM-контроллер может выполнить некоторое, ограниченное сверху, количество операций с банком памяти
        прежде, чем определится с игровым ходом. Если обученная DDPG будет использовать такую возможность,
        значит использование внешнего банка памяти дает преимущество при выборе игрового хода.

        Начальное заполнение банка памяти ? [1, 2, 3, ...] ?

        :return: None
        """

        def check_of_end_game(game_state: list[list[TicTacToePosStatus]]) -> TicTacToeGameStatus:
            transpose_game_state: list[list[TicTacToePosStatus]] = [list(x) for x in zip(*game_state)]

            if [TicTacToePosStatus.CROSS] * 3 in game_state \
                    or [TicTacToePosStatus.CROSS] * 3 in transpose_game_state \
                    or (game_state[0][0] == TicTacToePosStatus.CROSS
                        and game_state[1][1] == TicTacToePosStatus.CROSS
                        and game_state[2][2] == TicTacToePosStatus.CROSS) \
                    or (transpose_game_state[0][0] == TicTacToePosStatus.CROSS
                        and transpose_game_state[1][1] == TicTacToePosStatus.CROSS
                        and transpose_game_state[2][2] == TicTacToePosStatus.CROSS):
                return TicTacToeGameStatus.CROSS_WON
            elif [TicTacToePosStatus.ZERO] * 3 in game_state \
                    or [TicTacToePosStatus.ZERO] * 3 in transpose_game_state \
                    or (game_state[0][0] == TicTacToePosStatus.ZERO
                        and game_state[1][1] == TicTacToePosStatus.ZERO
                        and game_state[2][2] == TicTacToePosStatus.ZERO) \
                    or (transpose_game_state[0][0] == TicTacToePosStatus.ZERO
                        and transpose_game_state[1][1] == TicTacToePosStatus.ZERO
                        and transpose_game_state[2][2] == TicTacToePosStatus.ZERO):
                return TicTacToeGameStatus.ZERO_WON
            elif TicTacToePosStatus.EMPTY in itertools.chain(*game_state):
                return TicTacToeGameStatus.THE_GAME_CONTINUES
            else:
                return TicTacToeGameStatus.DRAW

        def opponent_action(game_state: list[list[TicTacToePosStatus]]) \
                -> Point2D | None:
            """
            Реализация стратегии опонента в виде случайного выбора позиции
            :param game_state: game state
            :return: Позиция хода опонента (row, column)
            """
            empty_pos: [tuple[int, int]] = [(r, c) for r in range(3) for c in range(3)
                                            if game_state[r][c] == TicTacToePosStatus.EMPTY]
            if empty_pos:
                return empty_pos[random.randrange(0, len(empty_pos))]
            else:
                return None

        def game_state_to_model_input(game_state: list[list[TicTacToePosStatus]]) -> Tensor:
            return tf.reshape(
                tensor=tf.constant(list(map(lambda x: x.value, itertools.chain(*game_state))),
                                   dtype=tf.float32),
                shape=(3 * 3)
            )

        def get_actor_dense():
            last_init = tf.random_uniform_initializer(minval=-1E-0, maxval=1E-0)

            # !!! не используем layers.BatchNormalization(), так как игра меняеется в ходе обучения Actor и
            # из-за этого смещается распределение входных сигналов. Это создает нестабильность восприятия игровой среды.

            inputs = layers.Input(shape=(3 * 3,))
            out = layers.Dense(10, activation="softsign", kernel_initializer=last_init)(inputs)
            out = layers.Dense(10, activation="softsign", kernel_initializer=last_init)(out)
            outputs = layers.Dense(9, activation="linear", kernel_initializer=last_init)(out)

            model = tf.keras.Model(inputs, outputs)
            return model

        def get_critic_dense():
            last_init = tf.random_uniform_initializer(minval=-1E-0, maxval=1E-0)

            # !!! не используем layers.BatchNormalization(), так как игра меняеется в ходе обучения Actor и
            # из-за этого смещается распределение входных сигналов. Это создает нестабильность восприятия игровой среды.

            inputs = layers.Input(shape=(3 * 3 + 9,))
            out = layers.Dense(10, activation="softsign", kernel_initializer=last_init)(inputs)
            out = layers.Dense(10, activation="softsign", kernel_initializer=last_init)(out)
            outputs = layers.Dense(1, activation="linear", kernel_initializer=last_init)(out)

            model = tf.keras.Model(inputs, outputs)

            return model

        target_actor = DenseModel(get_actor_dense(), 9)
        actor_model = DenseModel(get_actor_dense(), 9)
        target_critic = DenseModel(get_critic_dense(), 1)
        critic_model = DenseModel(get_critic_dense(), 1)

        # !!! weight_decay=1.0 Предотвращаем попадание входа функции активации в область "насыщения".
        critic_optimizer: Optimizer = tf.keras.optimizers.Adam(learning_rate=1E-3, weight_decay=1.0)
        actor_optimizer: Optimizer = tf.keras.optimizers.Adam(learning_rate=1E-3, weight_decay=1.0)

        # содаем буфер обучающих примеров
        buffer: Buffer = Buffer(buffer_capacity=50000)

        # создаем внешний банк памяти из 10 ячеек с 5 позициями.
        memory_number_of_cells: int = 10
        memory_cell_length: int = 5
        memory_bank: MemoryBank = MemoryBank(
            memory_bank_buffer=tuple(tf.Variable(initial_value=[0] * memory_cell_length, dtype=tf.float32) for _
                                     in range(memory_number_of_cells)),
            memory_cell_length=memory_cell_length
        )
        # тензор весов для локализации чтения и записи
        memory_bank_w: tf.Variable = tf.Variable(initial_value=[0] * memory_number_of_cells, dtype=tf.float32,
                                                 trainable=False)
        # удаления
        memory_bank_e: tf.Variable = tf.Variable(initial_value=[0] * memory_cell_length, dtype=tf.float32,
                                                 trainable=False)
        # добавления
        memory_bank_a: tf.Variable = tf.Variable(initial_value=[0] * memory_cell_length, dtype=tf.float32,
                                                 trainable=False)
        # сохранения считанных данных
        memory_bank_r: tf.Variable = tf.Variable(initial_value=[0] * memory_cell_length, dtype=tf.float32,
                                                 trainable=False)
        # интерполяции
        memory_bank_interpolation_gate: tf.Variable = tf.Variable(initial_value=[0], dtype=tf.float32, trainable=False)
        # сдвига
        memory_bank_distribution_shifts: tf.Variable = tf.Variable(initial_value=[0] * (memory_cell_length * 2 - 1),
                                                                   dtype=tf.float32, trainable=False)
        # фокусировки
        memory_bank_focus_factor: tf.Variable = tf.Variable(initial_value=[0], dtype=tf.float32, trainable=False)

        ddpg: DDPG = DDPG(
            target_critic=target_critic,
            target_actor=target_actor,
            critic_model=critic_model,
            actor_model=actor_model,
            critic_model_input_concat=lambda state_batch, action_batch:
            tf.concat(values=[state_batch, action_batch], axis=1)
        )

        exploration_noise: OUActionNoise = OUActionNoise(mean=tf.constant([0.0] * 9),
                                                         std_deviation=tf.constant([1.0] * 9))

        # Разыгрываем серию игр.
        # DDPG играет за кресты против условного игрока, который случайным образом заполняет нулями свободные ячейки.
        # Для каждой следующий 100 игр считаем % побед DDPG.
        # Состояним среды для DDPG является текущее состояние игрового поля.
        # Выход DDPG дополнительно обрабатывается функцией softmax.
        # Затем выбирается свободная позиция с максимальным значением вероятности.
        # В эту позицию DDPG помещает свой знак.
        win_rate: float = 0
        history_win_rate: list[float] = []
        game_state: list[list[TicTacToePosStatus]] = \
            [[TicTacToePosStatus.EMPTY] * 3, [TicTacToePosStatus.EMPTY] * 3, [TicTacToePosStatus.EMPTY] * 3]
        for game_num in range(100000):
            for r in range(3):
                for c in range(3):
                    game_state[r][c] = TicTacToePosStatus.EMPTY

            while True:
                original_action: Tensor = ddpg.policy(game_state_to_model_input(game_state))
                k: Tensor = tf.reduce_mean(tf.abs(original_action))
                noise: Tensor = exploration_noise() * (k * 0.2 if k > 0 else 1)
                original_action = original_action + noise

                action: Tensor = tf.reshape(tensor=tf.keras.activations.softmax(
                    tf.reshape(tensor=original_action, shape=(1, 3 * 3))),
                    shape=(3, 3))

                # выбираем свободную позицию для хода
                point: Point2D | None = None
                max_prob: float = -1
                for r in range(3):
                    for c in range(3):
                        if game_state[r][c] == TicTacToePosStatus.EMPTY:
                            prob: float = float(action[r, c])
                            if prob > max_prob:
                                point = Point2D(r, c)
                                max_prob = prob

                # ход DDPG
                prev_game_state = deepcopy(game_state)
                if point:
                    game_state[point.r][point.c] = TicTacToePosStatus.CROSS
                else:
                    self.fail(msg="Game solution not selected")

                # проверяем статус игры
                game_status: TicTacToeGameStatus = check_of_end_game(game_state)
                reward: float = 0
                if game_status == TicTacToeGameStatus.CROSS_WON:
                    reward = 1

                if game_status == TicTacToeGameStatus.THE_GAME_CONTINUES:
                    r, c = opponent_action(game_state)
                    game_state[r][c] = TicTacToePosStatus.ZERO

                    # проверяем статус игры
                    game_status = check_of_end_game(game_state)

                    if game_status == TicTacToeGameStatus.ZERO_WON:
                        reward = -1

                buffer.record(state=game_state_to_model_input(prev_game_state),
                              action=original_action,
                              reward=reward,
                              next_state=game_state_to_model_input(game_state))

                ddpg.learn(buffer=buffer,
                           batch_size=10,
                           epochs=5,
                           q_learning_discount_factor=0.9,
                           target_model_update_rate=0.1,
                           critic_optimizer=critic_optimizer,
                           actor_optimizer=actor_optimizer
                           )

                if game_status != TicTacToeGameStatus.THE_GAME_CONTINUES:
                    # собираем статистику по исходам игры
                    if game_status == TicTacToeGameStatus.CROSS_WON:
                        win_rate += 1

                    if game_num > 0 and game_num % 100 == 0:
                        history_win_rate.append(win_rate / 100)
                        win_rate = 0
                        print(sum(history_win_rate) / len(history_win_rate))
                        print(sum(history_win_rate[len(history_win_rate) - 30:]) / len(
                            history_win_rate[len(history_win_rate) - 30:]))
                        print(history_win_rate[len(history_win_rate) - 10:])
                        print(len(history_win_rate))

                    break

        print(history_win_rate)


    def test_learn(self):
        # self._test_tic_tac_toe_without_ntm()
        self._test_tic_tac_toe_mlp_ntm()
