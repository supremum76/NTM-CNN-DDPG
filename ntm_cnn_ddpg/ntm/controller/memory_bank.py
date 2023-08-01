"""
Controller of Neural Turing Machines.
See NTM_
.. _NTM: https://arxiv.org/pdf/1410.5401.pdf
"""

from typing import Callable, Sequence

import tensorflow as tf

MutableTensor = tf.Variable
ImmutableTensor = tf.Tensor
Tensor = ImmutableTensor | MutableTensor

# (shape, initial_value) -> Tensor
TensorFactory = Callable[[Sequence, Sequence], MutableTensor]

# Mutable tensor for storing data of memory bank cell
MemoryCell = tf.TensorSpec(shape=[None], dtype=tf.float32, name="MemoryCell")
MemoryBankBuffer = tuple[MemoryCell, ...]


# TODO Сделать все методы принимающими mutable-тензор для записи результата.
#  Все внутренние тензоры-буферы создать в init.
#  Все методы пометить как @tf.function
#  tensor_factory не нужна.

# def default_tensor_factory(shape: Sequence, initial_value: Sequence) -> MutableTensor:
#    return tf.Variable(initial_value=initial_value,
#                       shape=shape, dtype=tf.dtypes.float32, trainable=False)


class MemoryBank:
    def __init__(self, memory_bank_buffer: MemoryBankBuffer, memory_cell_length: int) -> None:
        """

        :param memory_bank_buffer: Mutable tensor for storing data of memory bank
        :memory_cell_length:
        :param tensor_factory:
        """
        self._memory_bank: MemoryBankBuffer = memory_bank_buffer
        self.memory_number_of_cells: int = len(memory_bank_buffer)
        self.memory_cell_length: int = memory_cell_length

        # вспомогательный буфер формы (1), для использования в вычислениях
        self._buffer_shape1: MutableTensor = tf.Variable(initial_value=[0], dtype=tf.dtypes.float32, shape=1)

    @tf.function
    def reading(self, w: Tensor, read_buffer: MutableTensor) -> None:
        """
        Чтение банка памяти.
        :param read_buffer: A mutable tensor for storing read data
        :param w: Вектор весов чтения ячеек банка памяти. Длина вектора должна равняться количеству ячеек банка памяти.
        Значения весов должны быть неотрицательными. Сумма весов должна быть равна единице.
        :return: Считанный вектор. Длина вектора равна длине ячейки банка памяти.
        """
        read_buffer.assign(value=[0]*self.memory_cell_length, read_value=False)
        for i in range(self.memory_number_of_cells):
            read_buffer.assign_add(w[i] * self._memory_bank[i], read_value=False)

    @tf.function
    def writing(self, w: Tensor, e: Tensor, a: Tensor) -> None:
        """
        Запись в банк памяти.

        Применена модификация оригинального алгоритма.
        В оригинальном алгоритме
            1) m := m * (1 - w[i] * e)
            2) m := m + w[i] * a
        Это может приводить к ошибке переполнения.
        Здесь алгоритм во втором шаге изменен
        m := m + w[i] * e * a
        Блок памяти по прежнему остается универсальным, так как любой код можно закодировать последовательностью из 2-х
        и более знаков из фиксированного словаря. Здесь же словарь знаков даже не фиксирован
        (доступно все множество рациональных чисел), но только их значение ограничено значениями вектора для добавления.
        :param w:
        :param e:
        :param a:
        """
        for i in range(self.memory_number_of_cells):
            f: MutableTensor = e * w[i]
            # erase
            self._memory_bank[i].assign(self._memory_bank[i] * (1 - f), read_value=False)
            # add
            self._memory_bank[i].assign_add(f * a, read_value=False)

    @tf.function
    def focusing(self, w_previous: Tensor, key_content: Tensor, interpolation_gate: Tensor, focus_factor: Tensor,
                 distribution_shifts: Tensor, w_next: MutableTensor) -> None:

        # Focusing by Content
        w_next_c: ImmutableTensor = tf.reshape(tensor=tf.keras.activations.softmax(
            tf.reshape(tensor=tf.vectorized_map(fn=lambda x: tf.math.exp(focus_factor *
                                                                         self._cosine_similarity(key_content, x)),
                                                elems=tf.stack(values=self._memory_bank, axis=0)),
                       shape=(1, self.memory_number_of_cells))),
            shape=[self.memory_number_of_cells])

        # interpolation
        one: ImmutableTensor = tf.constant(value=1, dtype=tf.dtypes.float32)
        w_next_i: ImmutableTensor = interpolation_gate * w_next_c + (one - interpolation_gate) * w_previous

        # Focusing by Location
        # shift
        tensor_list: list[float] = []
        s: MutableTensor = self._buffer_shape1
        zero: ImmutableTensor = tf.constant(value=0, dtype=tf.dtypes.float32, shape=[1])
        for i in range(self.memory_number_of_cells):
            s.assign(zero, read_value=False)
            # начинаем перечислять индекс в диапазоне
            # [-self.memory_number_of_cells + 1 , self.memory_number_of_cells - 1)
            # чтобы заколцевать ленту памяти при вычислении сдвига
            for j in range(-self.memory_number_of_cells + 1, self.memory_number_of_cells):
                s.assign_add(tf.reshape(tensor=w_next_i[j] * distribution_shifts[(i - j)],
                                        shape=[1]),
                             read_value=False)
            tensor_list.append(s.value())

        w_next.assign(tf.reshape(tensor=tf.stack(values=tensor_list), shape=[self.memory_number_of_cells]))

    @staticmethod
    @tf.function
    def _cosine_similarity(a: Tensor, b: Tensor) -> ImmutableTensor:
        return tf.reduce_sum(a * b) / (tf.norm(a) * tf.norm(b) + tf.keras.backend.epsilon())
