import collections
import gc
import itertools
import random
from collections import namedtuple
from copy import deepcopy
from enum import Enum
from typing import Iterator, Sequence, Any
from unittest import TestCase

import tensorflow as tf
from keras import layers
from keras.optimizers import Optimizer

from ntm_cnn_ddpg.cnn.model import Tensor, Model, ActorModel as DDPGActorModel, CriticModel as DDPGCriticModel, \
    OptionalSeqTensors
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


# TODO попробовать categorical policy (смотри tf.distributions.Categorical или tf.multinomial )
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

        exploration_noise: OUActionNoise = OUActionNoise(mean=tf.constant([0.0] * 9),
                                                         std_deviation=tf.constant([1.0] * 9))

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
                        print(sum(history_win_rate) / len(history_win_rate))
                        print(sum(history_win_rate[len(history_win_rate) - 30:]) / len(
                            history_win_rate[len(history_win_rate) - 30:]))
                        print(history_win_rate[len(history_win_rate) - 10:])
                        print(len(history_win_rate))

                    break

        print(history_win_rate)

    # TODO
    #   1) TEST 1 проверить  при base_block > 2 и точной истории последовательности глубиной base_block * 2
    #   2) TEST 2 заменить точную историю на ESN с количеством нейронов base_block * 0.1, base_block * 0.5, ...
    def _test_binary_sequence_prediction_mlp(self) -> None:
        """
            Проверяем связку DDPG+MLP на "игре", требующей памяти.
            Например, предсказание бинарной последовательности, являющейся
            периодическим повторением некоторой однократно случайно сформированной бинарной последовательности заданной
            длины. Без использования памяти прогноз такой последовательности невозможен.
            на входе актора текущий считанный бит и считанная память, на выходе прогноз значения следующего бита.
            Примеры:
            01 01 01 0 -> 1
            100 100 10 -> 0
            1101 1101 1 -> 1

            Необходимо оценить зависимость кол-ва итераций обучения для достижения точности 90% от длины базовой
            последовательности. Остальные параметры фиксируем.
            Можно вывести эмприческую аппроксиацию зависимости.

            :return: None
            """

        # TODO попробовать в качестве памяти использовать ART1 или ART2
        # TODO попробовать в качестве памяти использовать блок напрямую управляемый агентом.
        #       Одна ячейка такой памяти соответствует одному выходу нейросети контроллера.
        base_block = [0] * 25 + [1] * 25
        random.shuffle(base_block)
        base_block_length: int = len(
            base_block)  # base_block_length < 100. При 100 агент не обучался, заводя поток примеров в тупик
        memory_size: int = base_block_length

        actor_output_length: int = (
            2  # прогноз последовательности (0 или 1)
        )

        def get_actor_dense():
            last_init = tf.random_uniform_initializer(minval=-1E-0, maxval=1E-0)

            inputs = layers.Input(shape=memory_size)
            out = layers.Dense(base_block_length, activation="relu", kernel_initializer=last_init)(inputs)
            out = layers.Dense(base_block_length, activation="relu", kernel_initializer=last_init)(out)
            outputs = layers.Dense(actor_output_length, activation="linear", kernel_initializer=last_init)(out)

            model = tf.keras.Model(inputs, outputs)
            return model

        def get_critic_dense():
            last_init = tf.random_uniform_initializer(minval=-1E-0, maxval=1E-0)

            inputs = layers.Input(
                shape=(
                        memory_size  # текущее состояние среды
                        + actor_output_length  # текущее действие актора
                )
            )
            out = layers.Dense(base_block_length, activation="relu", kernel_initializer=last_init)(inputs)
            out = layers.Dense(base_block_length, activation="relu", kernel_initializer=last_init)(out)
            outputs = layers.Dense(1, activation="linear", kernel_initializer=last_init)(out)

            model = tf.keras.Model(inputs, outputs)

            return model

        # содаем буфер обучающих примеров
        buffer: Buffer = Buffer(buffer_capacity=100000)

        target_actor = DenseModel(get_actor_dense(), actor_output_length)
        actor_model = DenseModel(get_actor_dense(), actor_output_length)
        target_critic = DenseModel(get_critic_dense(), 1)
        critic_model = DenseModel(get_critic_dense(), 1)

        # !!! weight_decay=1.0 Предотвращаем попадание входа функции активации в область "насыщения".
        critic_optimizer: Optimizer = tf.keras.optimizers.Adam(learning_rate=1E-3, weight_decay=1.0)
        actor_optimizer: Optimizer = tf.keras.optimizers.Adam(learning_rate=1E-3, weight_decay=1.0)

        ddpg: DDPG = DDPG(
            target_critic=target_critic,
            target_actor=target_actor,
            critic_model=critic_model,
            actor_model=actor_model,
            critic_model_input_concat=lambda state_batch, action_batch:
            tf.concat(values=[state_batch, action_batch], axis=1)
        )

        exploration_noise: OUActionNoise = OUActionNoise(mean=tf.constant([0.0] * actor_output_length),
                                                         std_deviation=tf.constant([1.0] * actor_output_length))

        # Генератор бинарной последовательности
        def binary_sequence_generator(sequence_length: int) -> Iterator[int]:
            for i in range(sequence_length):
                yield base_block[i % base_block_length]

        accuracy: float = 0
        history_accuracy: list[float] = []
        avg_reward: float = 0
        noise_level: float = 0.2

        binary_sequence: Iterator[int] = iter(binary_sequence_generator(100000))
        step: int = 0
        bit: int = next(binary_sequence)
        memory = collections.deque([0] * (memory_size - 1), maxlen=memory_size)
        memory.append(bit)
        while True:
            original_action: Tensor = ddpg.policy(state=tf.constant(memory, dtype=tf.float32))

            #  разделяем действие
            #  добавлять шум исследования после разделения действия
            index_from, index_to = 0, 2
            forecast = original_action[index_from:index_to]
            k = tf.reduce_mean(tf.abs(forecast))
            noise: Tensor = exploration_noise()
            noise_forecast = forecast + noise[index_from:index_to] * (k * noise_level if k > 0 else 1)

            forecast = tf.reshape(
                tensor=tf.keras.activations.softmax(tf.reshape(tensor=noise_forecast, shape=(1, 2))),
                shape=2
            )
            next_bit_prediction: int = tf.argmax(forecast).numpy()
            next_zero_prediction_prob = forecast[0].numpy()

            try:
                bit = next(binary_sequence)
            except StopIteration:
                break

            prev_memory = memory.copy()
            memory.popleft()
            memory.append(bit)

            reward: float = \
                next_zero_prediction_prob - 0.5 if bit == 0 else (1 - next_zero_prediction_prob) - 0.5
            avg_reward += reward

            buffer.record(
                state=tf.constant(prev_memory, dtype=tf.float32),
                action=noise_forecast,
                reward=reward,
                next_state=tf.constant(value=memory, dtype=tf.float32))

            ddpg.learn(buffer=buffer,
                       batch_size=1000,
                       epochs=5,
                       q_learning_discount_factor=0.9,
                       target_model_update_rate=0.1,
                       critic_optimizer=critic_optimizer,
                       actor_optimizer=actor_optimizer
                       )

            step += 1

            # сбор статистики точности прогнозирования
            if bit == next_bit_prediction:
                accuracy += 1

            if step % 100 == 0:
                history_accuracy.append(accuracy / 100)
                accuracy = 0

                avg_reward /= 100
                print(avg_reward)
                avg_reward = 0

                print(sum(history_accuracy) / len(history_accuracy))
                print(sum(history_accuracy[len(history_accuracy) - 30:]) / len(
                    history_accuracy[len(history_accuracy) - 30:]))
                print(history_accuracy[len(history_accuracy) - 10:])
                print(len(history_accuracy))

    def _test_binary_sequence_prediction_mlp_lstm(self) -> None:
        """
            Проверяем связку DDPG+MLP+LSTM на "игре", требующей памяти.
            Например, предсказание бинарной последовательности, являющейся
            периодическим повторением некоторой однократно случайно сформированной бинарной последовательности заданной
            длины. Без использования памяти прогноз такой последовательности невозможен.
            на входе актора ограниченная история бинарной последовательности,
            на выходе прогноз значения следующего бита.
            Примеры:
            01 01 01 0 -> 1
            100 100 10 -> 0
            1101 1101 1 -> 1


            Необходимо оценить зависимость кол-ва итераций обучения для достижения точности 90% от длины базовой
            последовательности. Остальные параметры фиксируем.
            Можно вывести эмприческую аппроксиацию зависимости.

            :return: None
        """

        base_block = [0] * 2 + [1] * 2  # [0] * 5 + [1] * 5
        random.shuffle(base_block)
        base_block_length: int = len(base_block)
        # История бинарной последовательности ограниченным окном.
        # Проверяем, что LSTM способна прогнозировать такие последовательности.
        sequence_history_window: collections.deque = collections.deque([], base_block_length * 10)
        sequence_history_window_extended: collections.deque = collections.deque([], base_block_length * 10)
        lstm_units: int = int(base_block_length * 1.0)  # base_block_length * 2; 0.5 - нет обучения
        lstm_dropout: float = 0.0  # Значения отличные от 0.0 приводят к значительному увеличению потребляемой памяти
        batch_size: int = 20 * base_block_length
        actor_optimizing_start_step = 5 * batch_size
        buffer_capacity = 500 * base_block_length

        actor_output_length: int = (
            2  # прогноз последовательности (0 или 1)
        )

        @tf.function
        def activation1(x: Tensor):
            """
            x -> [-1, +1]
            :param x:
            :return: x -> [-1, +1]
            """
            return tf.math.sin(x)

        @tf.function
        def activation2(x: Tensor):
            """
            x -> [0, 1]
            :param x:
            :return: x -> [0, 1]
            """
            return tf.math.sin(x) ** 2

        class ActorModel(DDPGActorModel):
            def __init__(self):
                inputs = layers.Input(
                    shape=(
                        None,  # time
                        1  # за раз обрабатываем один бит из истории последовательности
                    )
                )

                lstm_out = (
                    tf.keras.layers.LSTM(units=lstm_units,
                                         activation=activation1,
                                         recurrent_activation=activation2,
                                         dropout=lstm_dropout,
                                         recurrent_dropout=lstm_dropout)(inputs))

                out = layers.Dense(lstm_units, activation="elu")(lstm_out)
                # out = layers.Dense(actor_output_length, activation=tf.keras.activations.softmax)(out)
                out = layers.Dense(actor_output_length, activation=activation1)(out)
                out = tf.keras.activations.softmax(out)
                model = tf.keras.Model(inputs=inputs, outputs=out)

                self.__model = model
                self.output_length = actor_output_length

            def predict(self, model_input: Tensor | Sequence[Tensor], training: bool, batch_mode: bool = True) -> \
                    Tensor | Sequence[Tensor]:
                result = (
                    self.__model(inputs=model_input if batch_mode else
                    # добавляем фиктивное измерение пакета
                    tf.reshape(tensor=model_input, shape=(1, *model_input.shape))
                    if not isinstance(model_input, Sequence)
                    else [tf.reshape(tensor=t, shape=(1, *t.shape)) for t in model_input],
                                 training=training))

                if not batch_mode:
                    # удаляем фиктивное измерение пакета
                    result = tf.reshape(tensor=result, shape=self.output_length)

                return result

            @property
            def trainable_variables(self) -> Tensor:
                return self.__model.trainable_variables

            def get_weights(self) -> Any:
                return [layer.get_weights() for layer in self.__model.layers if layer.count_params() > 0]

            def set_weights(self, weights: Any) -> None:
                for layer, weights in (
                        zip([layer for layer in self.__model.layers if layer.count_params() > 0], weights)):
                    layer.set_weights(weights)

        class ActorModel2(DDPGActorModel):
            def __init__(self):
                inputs = layers.Input(
                    shape=(
                        None,  # time
                        1  # за раз обрабатываем один бит из истории последовательности
                    )
                )

                lstm_initial_memory_state = layers.Input((lstm_units,))
                lstm_initial_carry_state = layers.Input((lstm_units,))

                lstm_out, lstm_final_memory_state, lstm_final_carry_state = (
                    tf.keras.layers.LSTM(units=lstm_units,
                                         activation=activation1,
                                         recurrent_activation=activation2,
                                         dropout=lstm_dropout,
                                         recurrent_dropout=lstm_dropout,
                                         return_state=True)
                    (inputs, initial_state=[lstm_initial_memory_state, lstm_initial_carry_state]))

                out = layers.Dense(lstm_units, activation="elu")(lstm_out)
                # out = layers.Dense(actor_output_length, activation=tf.keras.activations.softmax)(out)
                out = layers.Dense(actor_output_length, activation=activation1)(out)
                out = tf.keras.activations.softmax(out)

                model = tf.keras.Model(inputs=[inputs, lstm_initial_memory_state, lstm_initial_carry_state],
                                       outputs=[out, lstm_final_memory_state, lstm_final_carry_state])

                self.__model = model
                self.output_length = actor_output_length
                self.__lstm_memory_state = tf.constant([0] * lstm_units, dtype=tf.float32, shape=(1, lstm_units))
                self.__lstm_carry_state = tf.constant([0] * lstm_units, dtype=tf.float32, shape=(1, lstm_units))

            def predict(self, model_input: Tensor | Sequence[Tensor], training: bool, batch_mode: bool = True) -> \
                    Tensor | Sequence[Tensor]:

                online_mode: bool = not batch_mode

                lstm_init_memory_state: tf.Tensor = self.__lstm_memory_state
                lstm_init_carry_state: tf.Tensor = self.__lstm_carry_state

                # в пакетном режиме используем нулевое начальное состояние lstm-слоя
                if batch_mode:
                    batch_size: int = model_input[0].shape[0] if isinstance(model_input, Sequence) \
                        else model_input.shape[0]
                    lstm_batch_init_memory_state: tf.Tensor = (
                        tf.constant([0] * lstm_units * batch_size, dtype=tf.float32,
                                    shape=(batch_size, lstm_units)))

                    lstm_init_memory_state = lstm_batch_init_memory_state
                    lstm_init_carry_state = lstm_batch_init_memory_state

                result, lstm_final_memory_state, lstm_final_carry_state = (
                    self.__model(inputs=[model_input if batch_mode else
                                         # добавляем фиктивное измерение пакета
                                         tf.reshape(tensor=model_input, shape=(1, *model_input.shape))
                                         if not isinstance(model_input, Sequence)
                                         else [tf.reshape(tensor=t, shape=(1, *t.shape)) for t in model_input],
                                         lstm_init_memory_state,
                                         lstm_init_carry_state],
                                 training=training))

                if online_mode:
                    # удаляем фиктивное измерение пакета
                    result = tf.reshape(tensor=result, shape=self.output_length)
                    # запоминаем новое состояние lstm-слоя только в online-режиме
                    self.__lstm_memory_state = lstm_final_memory_state
                    self.__lstm_carry_state = lstm_final_carry_state

                return result

            @property
            def trainable_variables(self) -> Tensor:
                return self.__model.trainable_variables

        class CriticModel(DDPGCriticModel):
            def __init__(self):
                sequence_history = layers.Input(
                    shape=(
                        None,  # time
                        1  # за раз обрабатываем один бит из истории последовательности
                    )
                )

                action = layers.Input(
                    shape=actor_output_length  # текущее действие актора
                )

                out = tf.keras.layers.LSTM(
                    units=lstm_units,
                    activation=activation1,
                    recurrent_activation=activation2,
                    dropout=lstm_dropout,
                    recurrent_dropout=lstm_dropout)(
                    sequence_history)
                out = tf.keras.layers.Concatenate()([out, action])
                out = layers.Dense(lstm_units, activation="elu")(out)
                outputs = layers.Dense(1, activation=None)(out)

                model = tf.keras.Model([sequence_history, action], outputs)

                self.__model = model
                self.output_length = 1

            def predict(self, model_input: Tensor | Sequence[Tensor], training: bool,
                        batch_mode: bool = True) -> Tensor:
                return self.__model(inputs=model_input, training=training)

            @property
            def trainable_variables(self) -> Tensor:
                return self.__model.trainable_variables

            def get_weights(self) -> Any:
                return [layer.get_weights() for layer in self.__model.layers if layer.count_params() > 0]

            def set_weights(self, weights: Any) -> None:
                for layer, weights in (
                        zip([layer for layer in self.__model.layers if layer.count_params() > 0], weights)):
                    layer.set_weights(weights)

        def critic_model_input_concat(state_batch: OptionalSeqTensors, action_batch: OptionalSeqTensors) \
                -> OptionalSeqTensors:
            return list(itertools.chain.from_iterable([
                state_batch if isinstance(state_batch, Sequence) else [state_batch],
                action_batch if isinstance(action_batch, Sequence) else [action_batch]]))

        def optimal_action(critic_model: CriticModel, state_batch: OptionalSeqTensors) -> OptionalSeqTensors:
            action10 = tf.constant([[1.0, 0.0]])
            action01 = tf.constant([[0.0, 1.0]])

            def choice_action(state: OptionalSeqTensors) -> Tensor:
                reward10 = tf.reshape(critic_model.predict(critic_model_input_concat(
                    tf.reshape(tensor=state[0], shape=(1, *state[0].shape)), action10), training=False),
                    shape=())
                reward01 = tf.reshape(critic_model.predict(critic_model_input_concat(
                    tf.reshape(tensor=state[0], shape=(1, *state[0].shape)), action01), training=False),
                    shape=())
                comp = tf.cast(reward10 > reward01, dtype=tf.float32)
                return tf.math.reduce_sum(tf.stack([comp, 1 - comp], axis=0) *
                                          tf.stack([tf.reshape(action10, shape=[2]),
                                                    tf.reshape(action01, shape=[2])],
                                                   axis=0),
                                          axis=0)

            return tf.vectorized_map(fn=choice_action, elems=state_batch)

        # содаем буфер обучающих примеров
        buffer: Buffer = Buffer(buffer_capacity=buffer_capacity)

        target_actor = ActorModel()
        actor_model = ActorModel()
        target_critic = CriticModel()
        critic_model = CriticModel()

        # !!! weight_decay=1.0 Предотвращаем попадание входа функции активации в область "насыщения".
        critic_optimizer: Optimizer = tf.keras.optimizers.Adam(learning_rate=1E-4,  # 3
                                                               weight_decay=0.1,
                                                               global_clipnorm=1.0)  # global_clipnorm lr=0.01 weight_decay=0.1
        actor_optimizer: Optimizer = tf.keras.optimizers.Adam(learning_rate=1E-5,  # 4
                                                              weight_decay=0.1,  # 0.1
                                                              global_clipnorm=1.0)  # global_clipnorm lr=0.01 weight_decay=0.1

        ddpg: DDPG = DDPG(
            target_critic=target_critic,
            target_actor=target_actor,
            critic_model=critic_model,
            actor_model=actor_model,
            critic_model_input_concat=critic_model_input_concat
        )

        exploration_noise: OUActionNoise = OUActionNoise(mean=tf.constant([0.0] * actor_output_length),
                                                         std_deviation=tf.constant([0.2] * actor_output_length))

        # Генератор бинарной последовательности
        def binary_sequence_generator(sequence_length: int) -> Iterator[int]:
            for i in range(sequence_length):
                yield base_block[i % base_block_length]

        accuracy: float = 0
        history_accuracy: list[float] = []
        history_avg_reward: list[float] = []
        avg_reward: float = 0

        binary_sequence: Iterator[int] = iter(binary_sequence_generator(100000))
        # наполняем sequence_history_window
        for _ in range(sequence_history_window.maxlen):
            sequence_history_window.append(next(binary_sequence))
        step: int = 0
        bit: int = next(binary_sequence)
        sequence_history_window.append(bit)
        sequence_history_window_extended.append(bit)

        while True:
            original_action: Tensor = ddpg.policy(state=
                                                  tf.constant(value=sequence_history_window_extended, dtype=tf.float32,
                                                              shape=(len(sequence_history_window_extended), 1))
                                                  )

            #  добавляем шум исследования
            forecast = original_action
            noise: Tensor = exploration_noise()
            forecast = tf.reshape(
                tensor=tf.keras.activations.softmax(tf.reshape(tensor=(forecast + noise), shape=(1, 2))),
                shape=2
            )

            next_bit_prediction: int = tf.argmax(forecast).numpy()
            next_zero_prediction_prob = forecast[0].numpy()

            prev_state = sequence_history_window.copy()
            try:
                bit = next(binary_sequence)
                sequence_history_window.append(bit)
                sequence_history_window_extended.append(bit)
            except StopIteration:
                break

            reward: float = \
                next_zero_prediction_prob - 0.5 if bit == 0 else (1 - next_zero_prediction_prob) - 0.5

            # reward: float = \
            #    1.0 if bit == next_bit_prediction \
            #    else 2 * (next_zero_prediction_prob - 0.5 if bit == 0 else 0.5 - next_zero_prediction_prob)

            avg_reward += reward

            # представляем состояние (историю бинарной последовательности) как временную последовательность
            buffer.record(
                state=tf.constant(value=prev_state, dtype=tf.float32, shape=(len(prev_state), 1)),
                action=forecast,
                reward=reward,
                next_state=tf.constant(value=sequence_history_window, dtype=tf.float32,
                                       shape=(len(sequence_history_window), 1)),
            )

            ddpg.learn(buffer=buffer,
                       batch_size=batch_size,
                       epochs=1,
                       q_learning_discount_factor=0.9,  # q_learning_discount_factor=0.9
                       target_model_update_rate=0.001,
                       critic_optimizer=critic_optimizer,
                       # TODO ???
                       actor_optimizer=actor_optimizer if step > actor_optimizing_start_step else None,
                       optimal_action=optimal_action
                       )

            step += 1

            # сбор статистики точности прогнозирования
            if bit == next_bit_prediction:
                accuracy += 1

            if step % 100 == 0:
                avg_reward /= 100
                history_avg_reward.append(avg_reward)
                avg_reward = 0

                history_accuracy.append(accuracy / 100)
                accuracy = 0
                stat_window_len: int = 20
                # останавливаемся, если достигли предела качества управления
                if step > actor_optimizing_start_step + stat_window_len * 100 and step > 2 * stat_window_len * 100:
                    if not (sum(history_avg_reward[len(history_avg_reward) - stat_window_len:]) >
                            sum(history_avg_reward[len(history_avg_reward) -
                                                   2 * stat_window_len:len(history_avg_reward) - stat_window_len])):
                        print(f"stop! "
                              f"last avg reward {history_avg_reward[-1]} "
                              f"last avg reward {stat_window_len} {sum(history_avg_reward[len(history_avg_reward) - stat_window_len:]) / stat_window_len} "
                              f"last accuracy {history_accuracy[-1]} "
                              f"total steps {step}")
                        return

                if step > actor_optimizing_start_step and step > stat_window_len:
                    print(sum(history_avg_reward[len(history_avg_reward) - stat_window_len:]) / stat_window_len)
                    print(sum(history_accuracy[len(history_avg_reward) - stat_window_len:]) / stat_window_len)
                    print(history_avg_reward[len(history_avg_reward) - stat_window_len:])
                    print(history_accuracy[len(history_accuracy) - stat_window_len:])
                print(len(history_accuracy))

                gc.collect()

    def _test_binary_sequence_prediction_mlp_ntm(self) -> None:
        """
            Проверяем связку DDPG+MLP+NTM на "игре", требующей памяти.
            Например, предсказание бинарной последовательности, являющейся
            периодическим повторением некоторой однократно случайно сформированной бинарной последовательности заданной
            длины. Без использования памяти прогноз такой последовательности невозможен.
            на входе актора текущий считанный бит и считанная память, на выходе прогноз значения следующего бита.
            Примеры:
            01 01 01 0 -> 1
            100 100 10 -> 0
            1101 1101 1 -> 1

            Необходимо оценить зависимость кол-ва итераций обучения для достижения точности 90% от длины базовой
            последовательности. Остальные параметры фиксируем.
            Можно вывести эмприческую аппроксиацию зависимости.

            :return: None
            """

        base_block = [0] * 5 + [1] * 5
        random.shuffle(base_block)
        base_block_length: int = len(base_block)

        memory_number_of_cells: int = base_block_length
        memory_cell_length: int = base_block_length
        actor_output_length: int = (
                2  # прогноз последовательности (0 или 1)
                + memory_cell_length  # контент для фокусировки в блоке памяти для чтения/записи
                + memory_cell_length  # вектор удаления данных
                + memory_cell_length  # вектор записи данных
                + 2  # коэффициент интерполяции из диапазона [0, 1]
                + 1  # коэффициент контрастирования фокусировки
                # вектор сдвига чтения/записи =
                # [нулевой сдвиг, сдвиг на одну позицию вправо, сдвиг на одну позицию влево]
                + 3
        )

        def get_actor_dense():
            last_init = tf.random_uniform_initializer(minval=-1E-0, maxval=1E-0)

            inputs = layers.Input(shape=(
                    1  # текущий считанный бит последовательности
                    + memory_cell_length  # считанное из памяти при предыдущем действии актора
            ))
            out = layers.Dense(10, activation="softsign", kernel_initializer=last_init)(inputs)
            out = layers.Dense(10, activation="softsign", kernel_initializer=last_init)(out)
            outputs = layers.Dense(actor_output_length, activation="linear", kernel_initializer=last_init)(out)

            model = tf.keras.Model(inputs, outputs)
            return model

        def get_critic_dense():
            last_init = tf.random_uniform_initializer(minval=-1E-0, maxval=1E-0)

            inputs = layers.Input(
                shape=(
                    # текущее состояеие среды
                        1  # текущий считанный бит последовательности
                        + memory_cell_length  # считанное из памяти при предыдущем действии актора

                        + actor_output_length  # текущее действие актора
                )
            )
            out = layers.Dense(10, activation="softsign", kernel_initializer=last_init)(inputs)
            out = layers.Dense(10, activation="softsign", kernel_initializer=last_init)(out)
            outputs = layers.Dense(1, activation="linear", kernel_initializer=last_init)(out)

            model = tf.keras.Model(inputs, outputs)

            return model

        # содаем буфер обучающих примеров
        buffer: Buffer = Buffer(buffer_capacity=50000)

        # создаем внешний банк памяти
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
        prev_memory_bank_r: tf.Variable = tf.Variable(initial_value=[0] * memory_cell_length, dtype=tf.float32,
                                                      trainable=False)

        # интерполяции
        # shape = [2], чтобы можно было применять softmax и получать значения на отрезке [0, 1]
        memory_bank_interpolation_gate: tf.Variable = tf.Variable(initial_value=[0] * 2, dtype=tf.float32,
                                                                  trainable=False)
        # сдвига
        memory_bank_distribution_shifts: tf.Variable = tf.Variable(
            initial_value=[0] * (2 * memory_number_of_cells - 1),
            dtype=tf.float32,
            trainable=False)

        # фокусировки
        memory_bank_focus_factor: tf.Variable = tf.Variable(initial_value=[0], dtype=tf.float32, trainable=False)

        # контент
        memory_bank_key_content: tf.Variable = \
            tf.Variable(initial_value=[0] * memory_cell_length, dtype=tf.float32, trainable=False)

        target_actor = DenseModel(get_actor_dense(), actor_output_length)
        actor_model = DenseModel(get_actor_dense(), actor_output_length)
        target_critic = DenseModel(get_critic_dense(), 1)
        critic_model = DenseModel(get_critic_dense(), 1)

        # !!! weight_decay=1.0 Предотвращаем попадание входа функции активации в область "насыщения".
        critic_optimizer: Optimizer = tf.keras.optimizers.Adam(learning_rate=1E-3, weight_decay=1.0)
        actor_optimizer: Optimizer = tf.keras.optimizers.Adam(learning_rate=1E-3, weight_decay=1.0)

        ddpg: DDPG = DDPG(
            target_critic=target_critic,
            target_actor=target_actor,
            critic_model=critic_model,
            actor_model=actor_model,
            critic_model_input_concat=lambda state_batch, action_batch:
            tf.concat(values=[state_batch, action_batch], axis=1)
        )

        exploration_noise: OUActionNoise = OUActionNoise(mean=tf.constant([0.0] * actor_output_length),
                                                         std_deviation=tf.constant([1.0] * actor_output_length))

        # Генератор бинарной последовательности
        def binary_sequence_generator(sequence_length: int) -> Iterator[int]:
            for i in range(sequence_length):
                yield base_block[i % base_block_length]

        accuracy: float = 0
        history_accuracy: list[float] = []
        noise_level: float = 0.2

        # инициируем банк памяти
        i: int = 0
        e = tf.constant(value=[1] * memory_cell_length, dtype=tf.float32)
        for c in itertools.combinations_with_replacement([-1, 0, 1], memory_cell_length):
            w = tf.constant(value=[0 if j != i else 1 for j in range(memory_number_of_cells)],
                            dtype=tf.float32)
            a = tf.constant(value=c, dtype=tf.float32)
            memory_bank.writing(w=w, e=e, a=a)
            i += 1

        binary_sequence: Iterator[int] = iter(binary_sequence_generator(100000))
        step: int = 0
        bit: int = next(binary_sequence)
        while True:
            original_action: Tensor = ddpg.policy(
                tf.concat(values=(tf.constant([bit], dtype=tf.float32), memory_bank_r), axis=0)
            )

            #  разделяем действие
            #  добавлять шум исследования после разделения действия
            index_from, index_to = 0, 2
            forecast = original_action[index_from:index_to]
            k = tf.reduce_mean(tf.abs(forecast))
            noise: Tensor = exploration_noise()
            noise_forecast = forecast + noise[index_from:index_to] * (k * noise_level if k > 0 else 1)

            forecast = tf.reshape(
                tensor=tf.keras.activations.softmax(tf.reshape(tensor=noise_forecast, shape=(1, 2))),
                shape=2
            )
            next_bit_prediction: int = tf.argmax(forecast).numpy()
            next_zero_prediction_prob = forecast[0].numpy()

            index_from, index_to = index_to, index_to + memory_cell_length
            x = original_action[index_from:index_to]
            value_max = tf.reduce_max(x)
            k = tf.reduce_mean(tf.abs(x))
            x += noise[index_from:index_to] * (k * noise_level if k > 0 else 1)
            noise_memory_bank_e = x
            # приводим к [0, 1]
            value_max = tf.reduce_max(x)
            memory_bank_e.assign(x * tf.cast(x > 0, tf.float32) / (value_max + tf.keras.backend.epsilon()))

            index_from, index_to = index_to, index_to + memory_cell_length
            x = original_action[index_from:index_to]
            k = tf.reduce_mean(tf.abs(x))
            noise_memory_bank_a = x + noise[index_from:index_to] * (k * noise_level if k > 0 else 1)
            memory_bank_a.assign(noise_memory_bank_a)

            # memory_bank_interpolation_gate.shape = [2], для того чтобы можно было применить softmax
            # и в качестве итогового значения выбрать memory_bank_interpolation_gate[0]
            index_from, index_to = index_to, index_to + 2
            x = original_action[index_from:index_to]
            k = tf.reduce_mean(tf.abs(x))
            x += noise[index_from:index_to] * (k * noise_level if k > 0 else 1)
            noise_memory_bank_interpolation_gate = x
            memory_bank_interpolation_gate.assign(tf.reshape(
                tensor=tf.keras.activations.softmax(tf.reshape(tensor=x, shape=(1, 2))),
                shape=2))

            index_from, index_to = index_to, index_to + 1
            x = original_action[index_from:index_to]
            k = tf.reduce_mean(tf.abs(x))
            x += noise[index_from:index_to] * (k * noise_level if k > 0 else 1)
            noise_memory_bank_focus_factor = x
            memory_bank_focus_factor.assign(tf.clip_by_value(t=x * tf.cast(x > 0, tf.float32) + 1,
                                                             clip_value_min=1,
                                                             clip_value_max=20))

            index_from, index_to = index_to, index_to + 3
            x = original_action[index_from:index_to]
            k = tf.reduce_mean(tf.abs(x))
            x += noise[index_from:index_to] * (k * noise_level if k > 0 else 1)
            noise_memory_bank_distribution_shifts = x
            x = tf.reshape(
                tensor=tf.keras.activations.softmax(tf.reshape(tensor=x, shape=(1, index_to - index_from))),
                shape=index_to - index_from)
            memory_bank_distribution_shifts[0].assign(x[0])
            memory_bank_distribution_shifts[1].assign(x[1])
            memory_bank_distribution_shifts[-1].assign(x[-1])

            index_from, index_to = index_to, index_to + memory_cell_length
            x = original_action[index_from:index_to]
            k = tf.reduce_mean(tf.abs(x))
            x += noise[index_from:index_to] * (k * noise_level if k > 0 else 1)
            noise_memory_bank_key_content = x
            memory_bank_key_content.assign(x)

            memory_bank.focusing(w_previous=memory_bank_w,
                                 key_content=memory_bank_key_content,
                                 interpolation_gate=memory_bank_interpolation_gate[0],
                                 focus_factor=memory_bank_focus_factor,
                                 distribution_shifts=memory_bank_distribution_shifts,
                                 w_next=memory_bank_w)
            memory_bank.writing(w=memory_bank_w, a=memory_bank_a, e=memory_bank_e)
            prev_memory_bank_r.assign(memory_bank_r)
            memory_bank.reading(w=memory_bank_w, read_buffer=memory_bank_r)

            try:
                prev_bit: int = bit
                bit = next(binary_sequence)
            except StopIteration:
                break

            reward: float = \
                next_zero_prediction_prob - 0.5 if bit == 0 else (1 - next_zero_prediction_prob) - 0.5

            buffer.record(
                state=tf.concat(values=(tf.constant([prev_bit], dtype=tf.float32), prev_memory_bank_r), axis=0),
                action=tf.concat(
                    values=[
                        noise_forecast,
                        noise_memory_bank_e,
                        noise_memory_bank_a,
                        noise_memory_bank_interpolation_gate,
                        noise_memory_bank_focus_factor,
                        noise_memory_bank_distribution_shifts,
                        noise_memory_bank_key_content],
                    axis=0),
                reward=reward,
                next_state=tf.concat(values=(tf.constant([bit], dtype=tf.float32), memory_bank_r), axis=0))

            ddpg.learn(buffer=buffer,
                       batch_size=100,
                       epochs=5,
                       q_learning_discount_factor=0.9,
                       target_model_update_rate=0.1,
                       critic_optimizer=critic_optimizer,
                       actor_optimizer=actor_optimizer
                       )

            step += 1

            # сбор статистики точности прогнозирования
            if bit == next_bit_prediction:
                accuracy += 1

            if step % 100 == 0:
                history_accuracy.append(accuracy / 100)
                accuracy = 0
                print(sum(history_accuracy) / len(history_accuracy))
                print(sum(history_accuracy[len(history_accuracy) - 30:]) / len(
                    history_accuracy[len(history_accuracy) - 30:]))
                print(history_accuracy[len(history_accuracy) - 10:])
                print(len(history_accuracy))

    def test_learn(self):
        self._test_binary_sequence_prediction_mlp_lstm()
        # self._test_tic_tac_toe_without_ntm()
        # self._test_tic_tac_toe_mlp_ntm()
        # self._test_binary_sequence_prediction_mlp()
        # self._test_binary_sequence_prediction_mlp_ntm()
