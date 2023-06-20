from abc import ABC, abstractmethod

import tensorflow as tf

Tensor = tf.Tensor


class Model(ABC):
    @abstractmethod
    def predict(self, model_input: Tensor, training: bool) -> Tensor:
        """
        Фычисление функции модели.
        :param model_input: Входные данные модели в виде тензора формы (DIMENSIONS, FILTERS FOR INPUT) или,
        в случае вычислений для пакета входных данных, (BATCH SIZE, DIMENSIONS, FILTERS FOR INPUT).
        Здесь DIMENSIONS представляет скалярное значение, в случае 1-мерного входа модели, вектор составленный
        из взначений высоты и ширины, в случае 2-мерного входа модели и т.д.
        :param training: Признак использования результатов вычисления для обучения модели.
        :return: 1D тензор результатов вычисления функции модели в форме (LENGTH, FILTERS FOR OUTPUT) или,
        в случае вычислений для пакета входных данных, (BATCH SIZE, LENGTH, FILTERS FOR OUTPUT)
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
