import tensorflow as tf
from keras.models import Sequential as TensorFlowSequentialModel
from keras.layers import Layer, Input, Dense, Conv2D, Flatten, MaxPooling2D, BatchNormalization

Tensor = tf.Tensor


class Model2D:
    """
      2D CNN модель, слои которой флормируются по общему алгоритму.
      Слои группируются в модули, вычисления в которых огранизованы по схеме
      Batch Normalization -> Conv2D -> MaxPooling2D.
      Размер ядра свертки, размер окна объединения и количество фильтров задается для первого модуля.
      По мере автоматического добавления новых модулей, размер ядра свертки, размер окна объединения и
      количество фильтров увеличивается с приблизительно постоянной скоростью.
      Добавление модулей прекращается, когда размер выходного 2D тензора становится равным 1x1.
      После этого данные на финальном шаге обрабатываются полносвязной однослойной сетью.

      Одномерные входные тензоры преобразуются в набор двух-мерных тензоров.
      Для каждого компонента одномерного тензора создается один двух-мерный тензор, все компоненты которого
      устанавливаются в значение компонента одномерного тензора.

      Увеличение количества фильтров по мере добавления модулей преследует две цели.
      1) Решить проблему исчезающего градиента (vanishing gradient)
      2) Выстроить процесс вычислений так, что на первых уровнях из "сырых" данных
      выделяется оотносительно небольщой набор относительно простых признаков,
      а далее увеличивается количество выделяемых признаков, постепенно усложняя модель.

      Увеличение размера ядра свертки и размера окна объединения позволяет постепенно увеличивать сложность выделяемых
      признаков и постепенно уменьшать чувствительность к их позиции размещения.
    """

    __model: TensorFlowSequentialModel

    def __init__(self,
                 input_2d_shape: tuple[int, int, int],
                 input_1d_shape: tuple[int, int],
                 output_length: int,
                 start_filters: int,
                 start_kernel_size: tuple[int, int],
                 start_pool_size: tuple[int, int],
                 rate_of_filters_increase: float = 1,
                 rate_of_kernel_size_increase: float = 0,
                 rate_of_pool_size_increase: float = 0):
        """

        :param input_2d_shape: Форма 2D входа модели в формате (HEIGHT, WEIGHT, FILTERS FOR INPUT)
        :param input_1d_shape: Форма 1D входа модели в формате (LENGTH, FILTERS FOR INPUT)
        :param output_length: Длина выходного тензора модели
        :param start_filters: Количество фильтров в первом модуле модели
        :param start_kernel_size: Размер ядра свертки в первом модуле модели
        :param start_pool_size: Размер окна объединения в первом модуле модели
        :param rate_of_filters_increase: Коэффициент увеличения количества фильтров в каждом новом модуле модели по формуле
        F[i] = F[i-1] * ( 1 + rate_of_filters_increase)
        :param rate_of_kernel_size_increase: Коэффициент увеличения ядра свертки в каждом новом модуле модели по формуле
        K[i] = K[i-1] * ( 1 + rate_of_kernel_size_increase)
        :param rate_of_pool_size_increase: Коэффициент увеличения окна объединения в каждом новом модуле модели по формуле
        P[i] = P[i-1] * ( 1 + rate_of_pool_size_increase)
        """
        self.input_2d_shape = input_2d_shape
        self.input_1d_shape = input_1d_shape
        self.output_length = output_length
        self.start_filters = start_filters
        self.start_kernel_size = start_kernel_size
        self.start_pool_size = start_pool_size
        self.rate_of_filters_increase = rate_of_filters_increase
        self.rate_of_kernel_size_increase = rate_of_kernel_size_increase
        self.rate_of_pool_size_increase = rate_of_pool_size_increase
        self.__model = self.__model_build()

    def __model_build(self) -> TensorFlowSequentialModel:
        model = TensorFlowSequentialModel()

        input_shape: tuple[int, int, int] = (self.input_2d_shape[0], self.input_2d_shape[1],
                                             self.input_2d_shape[2] + self.input_1d_shape[0] * self.input_1d_shape[1])
        layer: Layer = Input(shape=input_shape, dtype=tf.float32)
        model.add(layer=layer)
        output_shape = layer.compute_output_shape(input_shape=input_shape)

        # выполняем цикл пока не получим на выходе сверточной сети тензор с формой (1, 1, filters).
        # Пока не сведем размер 2D плоскости к точке с некоторым количеством фильтров.
        # Применять свертку и объединение к такой точке уже не требуется.
        while output_shape[0] == 1 and output_shape[1] == 1:
            input_shape = output_shape
            # строим модуль, состоящий из последовательности слоев
            # слой нормализации
            # слой свертки
            # слой объединения
            output_shape = layer.compute_output_shape(input_shape=input_shape)
            break

        # выход модели окончательно вычисляет полносвязный слой персептрона с линейной функцией активации нейронов
        model.add(layer=Flatten())
        model.add(layer=BatchNormalization())
        model.add(layer=Dense(self.output_length, activation=None))

        return model

    def __1d_to_2d_input_tensor(self, tensor_1d: Tensor) -> Tensor:
        """
        Преобразует пакет одномерных (1D) многоканальных тензоров в
        пакет двумерных (2D) многоканальных тензоров.
        Форма (BATCH SIZE, LENGTH, FILTERS FOR INPUT) преобразуется к форме
        (BATCH SIZE, HEIGHT, WEIGHT, FILTERS FOR OUTPUT).
        Тип данных выходного тензора float32.

        Результат может быть использован как входные данные для 2D CNN-модели.

        Значения HEIGHT и WEIGHT берутся из формы входного 2D-тензора.
        FILTERS FOR OUTPUT = LENGTH * [FILTERS FOR INPUT].

        Каждый фильтр в выходном тензоре заполняется одним значением, взятым из соответсвующей позиции исходного
        одномерного тензора.

        :param tensor_1d: Тензор с формой (BATCH SIZE, LENGTH, FILTERS FOR INPUT) и числовым типом данных.
        :return: Тензор с формой (BATCH SIZE, HEIGHT, WEIGHT, FILTERS FOR OUTPUT) и типом данных float32,
        пригодный для обработки 2D CNN моделью.
        """
        batch_size: int = tensor_1d.shape[0]
        length: int = tensor_1d.shape[1]
        number_of_filters: int = tensor_1d.shape[2]

        return tf.reshape(
            tensor=tf.constant(value=[[[[[float(tensor_1d[b, v, f]) for v in range(length)]
                                         for f in range(number_of_filters)]
                                        for _ in range(self.input_2d_shape[1])]
                                       for _ in range(self.input_2d_shape[0])]
                                      for b in range(batch_size)],
                               shape=(batch_size,
                                      self.input_2d_shape[0], self.input_2d_shape[1],
                                      number_of_filters, length),
                               dtype=tf.dtypes.float32),
            shape=(batch_size, self.input_2d_shape[0], self.input_2d_shape[1], number_of_filters * length)
        )
