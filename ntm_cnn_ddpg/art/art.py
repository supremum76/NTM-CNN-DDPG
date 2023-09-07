"""
Реализация памяти, основанной на адаптивной резонансной теории [1].
Требуемые свойства памяти:
 1) Хранить информацию о прошлых состояниях среды с сохранением хронологического порядка.
 2) Выполнять сжатие информаци (возможно с потерей) в разряженный вектор,
    намного более меньший по сравнению с исходными данными.
 3) Память должна функционировать по неизменным универсальным алгоритмам в точности до небольшого количества
    гипер-параметров, не требующим точной настройки под конкретные задачи.

Сcылки:
    1) S. Grossberg, Adaptive pattern classification and universal recoding. II. Feedback, expectation, olfaction, and
    illusions, Bioi. Cybemet. 23, 1976, 187-202.
"""
import math

import tensorflow as tf
from tensorflow import Tensor


class ART:
    """
    Нейронная сеть, работающая по принципу адаптивной резонансной теории [1]. Архитектура сети в основном
    аналогична ахитектуре ART1 [2]. Но, в отличие от ART1, сеть позволяет классифицировать произвольные вектора
    (не только двоичные). В тоже время архитектура не связана с ART2/ART3, с точки зрения автора
    излишне усложненных для применения в практических задачах.

    Основные свойства:
     1) Для сравнения классифицируемых векторов с кодами распознавания используется косинусная мера близости векторов.
     2) Новый код распознавания инициируется значениями не классифицированного
        (слишком непохожего на все имеющиеся коды распознавания) вектора.
     3) Классифицируемые входные векторы никак не предобрабатываются. Контрастирование, нормализация,
        инвариантные преобразования и другое не применяются. При необходимости таких преобразований они
        должны быть выполненны перед подачей вектора на вход сети.
     4) Реализация позволяет менять размер входных классифицируемых векторов.

    Ограничения:
        Нулевые векторы не классифицируются.

    Сcылки:
    1) S. Grossberg, Adaptive pattern classification and universal recoding. II. Feedback, expectation, olfaction, and
    illusions, Bioi. Cybemet. 23, 1976, 187-202.
    2) G. A.Carpenter, S. Grossberg, A massively parallel architecture for a self-organizing neural pattern recognition
    machine, Computer Vision, Graphics, and Image Processing, Vol. 37, Issue 1,1987, 54-115.
    """

    recognition_codes: list[Tensor] = []  # коды распознавания
    recognition_codes_norm: list[Tensor] = []  # предрасчитанная норма кодов распознавания

    def __init__(self, init_input_size: int, similarity_threshold: float):
        """
        Инициирование ART-сети.
        :param init_input_size: Начальный размер входного классифицируемого вектора. В дальнейшем он может быть изменен.
        :param similarity_threshold: Пороговый коэффициент сходства входного вектора и кода распознавания.
        Должен находится в границах (0, 1). При значениях близких к 1 требуется почти полное сходство.
        """
        self.input_size = init_input_size
        self.similarity_threshold = similarity_threshold

    def __call__(self, *args, **kwargs) -> tuple[int, bool]:
        """
        Классифицирует входящий вектор и обучает ART-сеть.
        :param args: args[0] - ненулевой входящий вектор типа Tensor. Остальное игнорируется.
        :param kwargs: Игнорируется.
        :return: (номер класса, к которому был отнесен вектор или -1; признак добавления нового класса).
        Классы нумеруются с нуля.
        Если вместо номера класса возвращено -1, то значит вектор невозможно классифицировать.
        Единственная  возможная причина этого, на вход подан нулевой или близкий к нулю вектор.
        """
        _input: Tensor = args[0]
        # TODO проверить размер входящего вектора

        input_norm: Tensor = tf.norm(_input)
        if math.isclose(input_norm.numpy(), 0.0, rel_tol=1E-10, abs_tol=1E-10):
            return -1, False

        code_index: int = -1
        max_cosine_similarity: Tensor | None = None
        new_code: bool = False
        for i in range(len(self.recognition_codes)):
            code: Tensor = self.recognition_codes[i]
            code_norm: Tensor = self.recognition_codes_norm[i]
            cosine_similarity: Tensor = tf.reduce_sum(_input * code) / (input_norm * code_norm
                                                                        + tf.keras.backend.epsilon())
            if i > 0:
                if cosine_similarity > max_cosine_similarity:
                    max_cosine_similarity = cosine_similarity
                    code_index = i
            else:
                max_cosine_similarity = cosine_similarity
                code_index = 0

        if (code_index >= 0 and max_cosine_similarity.numpy() < self.similarity_threshold) or (code_index == -1):
            self.recognition_codes.append(tf.constant(value=_input))
            self.recognition_codes_norm.append(input_norm)
            code_index = len(self.recognition_codes) - 1
            new_code = True

        return code_index, new_code
