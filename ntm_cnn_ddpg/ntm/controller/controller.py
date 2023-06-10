import tensorflow as tf

Tensor = tf.Variable
MemoryBank = tf.TensorSpec(shape=[None, None], dtype=tf.float32, name="MemoryBank")


class Controller:
    def __init__(self, memory_bank: MemoryBank) -> None:
        self._memory_bank: MemoryBank = memory_bank
        self.memory_number_of_cells: int = memory_bank.shape[0]
        self.memory_cell_length: int = memory_bank.shape[1]

    def memory_bank_reading(self, w: Tensor, **kwargs) -> Tensor:
        """
        Чтение банка памяти.
        :param w: Вектор весов чтения ячеек банка памяти. Длина вектора должна равняться количеству ячеек банка памяти.
        Значения весов должны быть неотрицательными. Сумма весов должна быть равна единице.
        :param kwargs: Дополнительные параметры для конструктора возвращаемого тензора.
        :return: Считанный вектор. Длина вектора равна длине ячейки банка памяти.
        """
        result: Tensor = tf.Variable(initial_value=[0 for _ in range(self.memory_cell_length)],
                                     shape=self.memory_cell_length, dtype=tf.dtypes.float32, **kwargs)
        for i in range(w.shape[0]):
            result.assign_add(w[i] * self._memory_bank[i])

        return result

    def memory_bank_writing(self, w: Tensor, e: Tensor, a: Tensor) -> None:
        """
        Запись в банк памяти.
        :param w:
        :param e:
        :param a:
        """
        for i in range(w.shape[0]):
            # erase
            self._memory_bank[i].assign(self._memory_bank[i] * (1 - e * w[i]))
            # add
            self._memory_bank[i].assign_add(w[i] * a)

    def focusing(self, w_previous: Tensor, key_content: Tensor, interpolation_gate: Tensor, focus_factor: Tensor,
                 distribution_of_allowed_shifts: Tensor, **kwargs) -> Tensor:

        # Focusing by Content
        s: Tensor = tf.vectorized_map(
            fn=lambda x: tf.math.exp(focus_factor * self._cosine_similarity(key_content, x)),
            elems=self._memory_bank)

        w_next = tf.reshape(
            tensor=tf.vectorized_map(fn=lambda x: tf.math.exp(focus_factor *
                                                              self._cosine_similarity(key_content, x)) / s,
                                     elems=self._memory_bank),
            shape=self._memory_bank.shape[0]
        )

        # interpolation
        one: Tensor = tf.constant(1, dtype=tf.dtypes.float32)
        w_next = interpolation_gate * w_next + (one - interpolation_gate) * w_previous

        # Focusing by Location
        # shift
        tensor_list: list[Tensor] = []
        for i in range(self.memory_number_of_cells):
            s: Tensor = tf.constant(value=0, dtype=tf.dtypes.float32, shape=1)
            for j in range(self.memory_number_of_cells):
                s.assign_add(w_next[j] * distribution_of_allowed_shifts[(i - j) % self.memory_number_of_cells])
            tensor_list.append(s)

        w_next = tf.reshape(tensor=tf.Variable(initial_value=tensor_list,
                                               shape=(self.memory_number_of_cells, 1),
                                               dtype=tf.dtypes.float32, **kwargs),
                            shape=self.memory_number_of_cells)

        # final normalization (using the softmax function)
        w_next = tf.reshape(
            tensor=tf.keras.activations.softmax(tf.reshape(tensor=w_next, shape=(1, self.memory_number_of_cells))),
            shape=self.memory_number_of_cells)

        return w_next

    @staticmethod
    def _cosine_similarity(a: Tensor, b: Tensor) -> Tensor:
        return tf.reduce_sum(a * b) / (tf.norm(a) * tf.norm(b) + tf.keras.backend.epsilon())
