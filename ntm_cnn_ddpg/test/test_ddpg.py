import functools
import itertools
from enum import Enum
from unittest import TestCase

import tensorflow as tf
from keras.optimizers import Optimizer

from ntm_cnn_ddpg.cnn.model import Tensor
from ntm_cnn_ddpg.cnn.model2d import Model2D, concat_input_tensors
from ntm_cnn_ddpg.ddpg.ddpg import Buffer, DDPG, OUActionNoise


class TicTacToeGameStatus(Enum):
    DRAW = 0
    CROSS_WON = 1
    ZERO_WON = 2
    THE_GAME_CONTINUES = -1


class TicTacToePosStatus(Enum):
    EMPTY = 0
    CROSS = 1
    ZERO = -1


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

            if [TicTacToePosStatus.CROSS]*3 in game_state \
                    or [TicTacToePosStatus.CROSS]*3 in transpose_game_state \
                    or (game_state[0][0] == TicTacToePosStatus.CROSS
                        and game_state[1][1] == TicTacToePosStatus.CROSS
                        and game_state[2][2] == TicTacToePosStatus.CROSS) \
                    or (transpose_game_state[0][0] == TicTacToePosStatus.CROSS
                        and transpose_game_state[1][1] == TicTacToePosStatus.CROSS
                        and transpose_game_state[2][2] == TicTacToePosStatus.CROSS):
                return TicTacToeGameStatus.CROSS_WON
            elif [TicTacToePosStatus.ZERO]*3 in game_state \
                    or [TicTacToePosStatus.ZERO]*3 in transpose_game_state \
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

        game_state: list[list[TicTacToePosStatus]] = [[TicTacToePosStatus.EMPTY] * 3] * 3

        # создаем модели критика и актора
        target_actor: Model2D = Model2D(
            input_2d_shape=(3, 3, 1),
            input_1d_shape=(0, 0),
            output_length=3*3,
            start_filters=10,
            start_kernel_size=(2, 2),
            start_pool_size=(2, 2),
            rate_of_filters_increase=1,
            rate_of_kernel_size_increase=0,
            rate_of_pool_size_increase=0
        )

        target_critic: Model2D = Model2D(
            input_2d_shape=(3, 3, 1),
            input_1d_shape=(0, 0),
            output_length=3 * 3,
            start_filters=10,
            start_kernel_size=(2, 2),
            start_pool_size=(2, 2),
            rate_of_filters_increase=1,
            rate_of_kernel_size_increase=0,
            rate_of_pool_size_increase=0
        )

        actor_model: Model2D = Model2D(
            input_2d_shape=(3, 3, 1),
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
            input_2d_shape=(3, 3, 1),
            input_1d_shape=(0, 0),
            output_length=3 * 3,
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

        # содаем буфер и источник шума для исследований

        def critic_model_input_concat(state_batch: Tensor, action_batch: Tensor) -> Tensor:
            """
            :param state_batch: state batch
            :param action_batch: action batch
            :return: critic model input batch
            """

            return concat_input_tensors(state_batch, action_batch)

        buffer: Buffer = Buffer(
            buffer_capacity=10000,
            batch_size=100,
            target_critic=target_critic,
            target_actor=target_actor,
            critic_model=critic_model,
            actor_model=actor_model,
            critic_optimizer=critic_optimizer,
            actor_optimizer=actor_optimizer,
            q_learning_discount_factor=0.9,
            critic_model_input_concat=lambda state_batch, action_batch: concat_input_tensors(state_batch, action_batch))

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
        self.fail()

    def test_learn(self):
        self._test_tic_tac_toe_without_ntm()
