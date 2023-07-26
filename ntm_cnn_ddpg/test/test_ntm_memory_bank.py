import math
from unittest import TestCase

import tensorflow as tf

from ntm_cnn_ddpg.ntm.controller.memory_bank import MemoryBank, default_tensor_factory


class TestMemoryBank(TestCase):
    def test_memory_bank_writing_reading(self):
        memory_bank = MemoryBank(
            memory_bank_buffer=tuple(tf.Variable(initial_value=[0] * 5, dtype=tf.float32) for _ in range(10)),
            memory_cell_length=5
        )

        # writing
        w = tf.constant(value=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=tf.float32)
        e = tf.constant(value=[1] * 5, dtype=tf.float32)
        a = tf.constant(value=[1, 2, 3, 4, 5],
                        dtype=tf.float32)
        memory_bank.writing(w=w, e=e, a=a)

        w = tf.constant(value=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=tf.float32)
        e = tf.constant(value=[1] * 5, dtype=tf.float32)
        a = tf.constant(value=[-1, -2, -3, -4, -5], dtype=tf.float32)
        memory_bank.writing(w=w, e=e, a=a)

        w = tf.constant(value=[0, 0, 0, 0.3, 1, 0.7, 0, 0, 0, 0], dtype=tf.float32)
        e = tf.constant(value=[1] * 5, dtype=tf.float32)
        a = tf.constant(value=[10, 9, 7, 6, 5], dtype=tf.float32)
        memory_bank.writing(w=w, e=e, a=a)

        # rewriting
        w = tf.constant(value=[0.5, 0, 0, 0, 1, 0, 0, 0, 0, 0.5], dtype=tf.float32)
        e = tf.constant(value=[0.7, 0, 1, 0, 0.5], dtype=tf.float32)
        a = tf.constant(value=[-10, -9, -7, -6, -5], dtype=tf.float32)
        memory_bank.writing(w=w, e=e, a=a)

        # reading
        w = tf.constant(value=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=tf.float32)
        data = memory_bank.reading(w)
        test_data = tf.constant([1 * (1 - 0.7 * 0.5), 2, 3 * (1 - 1 * 0.5), 4, 5 * (1 - 0.5 * 0.5)]) + \
            tf.constant([-10 * 0.7 * 0.5, 0, -7 * 0.5, 0, -5 * 0.5 * 0.5])
        self.assertTrue(math.isclose(tf.reduce_sum(data).numpy(), tf.reduce_sum(test_data).numpy(),
                                     rel_tol=0.01, abs_tol=0.001))

        w = tf.constant(value=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=tf.float32)
        data = memory_bank.reading(w)
        test_data = tf.constant([-1 * (1 - 0.7 * 0.5), -2, -3 * (1 - 1 * 0.5), -4, -5 * (1 - 0.5 * 0.5)]) + \
            tf.constant([-10 * 0.7 * 0.5, 0, -7 * 0.5, 0, -5 * 0.5 * 0.5])
        self.assertTrue(math.isclose(tf.reduce_sum(data).numpy(), tf.reduce_sum(test_data).numpy(),
                                     rel_tol=0.01, abs_tol=0.001))

        w = tf.constant(value=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=tf.float32)
        data = memory_bank.reading(w)
        test_data = tf.constant([10 * (1 - 0.7), 9, 0, 6, 5 * (1 - 0.5)]) + \
            tf.constant([-10 * 0.7, 0, -7, 0, -5 * 0.5])
        self.assertTrue(math.isclose(tf.reduce_sum(data).numpy(), tf.reduce_sum(test_data).numpy(),
                                     rel_tol=0.01, abs_tol=0.001))

        w = tf.constant(value=[0, 0, 0, 0.5, 0, 0.5, 0, 0, 0, 0], dtype=tf.float32)
        data = memory_bank.reading(w)
        test_data = tf.constant([10, 9, 7, 6, 5], tf.float32) * 0.3 * 0.5 + tf.constant([10, 9, 7, 6, 5], tf.float32) \
            * 0.7 * 0.5
        self.assertTrue(math.isclose(tf.reduce_sum(data).numpy(), tf.reduce_sum(test_data).numpy(),
                                     rel_tol=0.01, abs_tol=0.001))

        w = tf.constant(value=[0, 0.2, 0.2, 0, 0, 0, 0.2, 0.2, 0.2, 0], dtype=tf.float32)
        data = memory_bank.reading(w)
        test_data = tf.constant([0, 0, 0, 0, 0], tf.float32)
        self.assertTrue(math.isclose(tf.reduce_sum(data).numpy(), tf.reduce_sum(test_data).numpy(),
                                     rel_tol=0.01, abs_tol=0.001))

    def test_focusing(self) -> None:
        # Проверяем фокусировку по точному совпадению ключа со сдвигом 1 и без интерполяции.

        memory_bank = MemoryBank(
            memory_bank_buffer=tuple(tf.Variable(initial_value=[0] * 5, dtype=tf.float32) for _ in range(10)),
            memory_cell_length=5
        )

        # writing
        w = tf.constant(value=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=tf.float32)
        e = tf.constant(value=[1] * 5, dtype=tf.float32)
        a = tf.constant(value=[1, 2, 3, 4, 5], dtype=tf.float32)
        memory_bank.writing(w=w, e=e, a=a)

        w = tf.constant(value=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=tf.float32)
        e = tf.constant(value=[1] * 5, dtype=tf.float32)
        a = tf.constant(value=[6, 7, 8, 9, 10], dtype=tf.float32)
        memory_bank.writing(w=w, e=e, a=a)

        # focusing
        w_previous = tf.constant(value=w)
        key_content = tf.constant(value=[1, 2, 3, 4, 5], dtype=tf.float32)
        interpolation_gate = tf.constant(1.0)
        focus_factor = tf.constant(20.0)
        distribution_of_allowed_shifts = tf.constant(value=[0, 1] + [0] * 18, dtype=tf.float32)
        w = memory_bank.focusing(w_previous=w_previous,
                                 key_content=key_content,
                                 interpolation_gate=interpolation_gate,
                                 focus_factor=focus_factor,
                                 distribution_of_allowed_shifts=distribution_of_allowed_shifts)

        # reading
        data = memory_bank.reading(w)

        self.assertTrue(math.isclose(tf.reduce_sum(data).numpy(), sum([6, 7, 8, 9, 10]), rel_tol=0.01, abs_tol=0.001))
