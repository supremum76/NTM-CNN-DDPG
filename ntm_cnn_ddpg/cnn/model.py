from abc import ABC, abstractmethod
from collections import Sequence

import tensorflow as tf

Tensor = tf.Tensor
OptionalSeqTensors = Tensor | Sequence[Tensor]
SeqTensors = Sequence[Tensor]


class Model(ABC):
    @abstractmethod
    def predict(self, model_input: OptionalSeqTensors, training: bool, batch_mode: bool = True) -> \
            OptionalSeqTensors:
        """
        Фычисление функции модели.
        :param model_input: Входные данные модели в виде одного тензора, если модель имеет один вход,
        либо последовательности тензоров, если модель имеет несколько входов.
        Здесь DIMENSIONS представляет скалярное значение, в случае 1-мерного входа модели, вектор составленный
        из взначений высоты и ширины, в случае 2-мерного входа модели и т.д.
        :param training: Признак использования результатов вычисления для обучения модели.
        :param batch_mode: Признак необходимости обработать пакет данных. Если флаг установлен, то в форме входных
        и выходных данных ожидается, что первоне измерение является измерением пакета.
        :return: Результатов вычисления функции модели.
        """
        pass

    @property
    @abstractmethod
    def trainable_variables(self) -> Tensor:
        """
        Возвразает тензор параметров модели, подлежащий коррекции в процессе обучения.
        :return: Обучаемые параметры модели.
        """
        pass


class ActorModel(Model, ABC):
    pass


class CriticModel(Model, ABC):
    """
    Особенность модели критика в том, что ее выход может быть только тензором.
    """
    @abstractmethod
    def predict(self, model_input: OptionalSeqTensors, training: bool, batch_mode: bool = True) -> Tensor:
        pass
