import functools
import itertools
from enum import Enum
from unittest import TestCase


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

    @staticmethod
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
        #  target_critic: Model
        #                  target_actor: Model,
        #                  critic_model: Model,
        #                  actor_model: Model,

        # содаем буфер и источник шума для исследований

        # Разыгрываем серию игр.
        # DDPG играет за кресты против условного игрока, который случайным образом заполняет нулями свободные ячейки.
        # Для каждой следующий 100 игр считаем % побед DDPG.
        # Состояним среды для DDPG является текущее состояние игрового поля.
        # Выход DDPG дополнительно обрабатывается функцией softmax.
        # Затем выбирается свободная позиция с максимальным значением вероятности.
        # В эту позицию DDPG помещает свой знак.

    def test_learn(self):
        self.fail()
