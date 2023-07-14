import tensorflow as tf
from keras.models import Sequential as TensorFlowSequentialModel
from keras.layers import Layer, Input, Dense, Conv2D, Flatten, MaxPooling2D, BatchNormalization
from tensorflow import TensorShape

from ntm_cnn_ddpg.cnn.model import Model, Tensor


def concat_input_tensors(input_2d: Tensor, input_1d: Tensor) -> Tensor:
    """
    Вспомогательная функция, позволяющая конкатенировать 2d и 1d входы модели в один 2d вход, с учетом
    специфики работы 2D CNN модели. Результат может быть использован как входные данные для 2D CNN-модели.

    Одномерная форма (LENGTH, FILTERS FOR INPUT) преобразуется к форме (HEIGHT, WEIGHT, FILTERS FOR OUTPUT).
    Значения HEIGHT и WEIGHT берутся из формы входного 2D-тензора.
    FILTERS FOR OUTPUT = LENGTH * [FILTERS FOR INPUT].
    Каждый фильтр в новом 2d тензоре заполняется одним значением, взятым из соответствующей
    позиции исходного 1d тензора.

    :param input_1d: Числовой тензор с формой (BATCH SIZE, LENGTH, FILTERS FOR INPUT) или (LENGTH, FILTERS FOR INPUT),
    в случае непакетной обработки.
    :param input_2d: Числовой тензор с формой (BATCH SIZE, HEIGHT, WEIGHT, FILTERS FOR INPUT)
    или (HEIGHT, WEIGHT, FILTERS FOR INPUT), в случае непакетной обработки.
    :return: Числовой (float32) тензор с формой (BATCH SIZE, HEIGHT, WEIGHT, FILTERS FOR OUTPUT)
    или (HEIGHT, WEIGHT, FILTERS FOR OUTPUT), , в случае непакетной обработки.
    Реультирующий тензор подходит для обработки 2D CNN моделью.
    """
    batch_size: int = input_1d.shape[0]
    length: int = input_1d.shape[1]
    number_of_filters: int = input_1d.shape[2]
    height: int = input_2d.shape[1]
    weight: int = input_2d.shape[2]

    _input_2d: Tensor
    _input_1d: Tensor
    batch_mode: bool = len(input_2d.shape) == 4
    if not batch_mode:
        # преобразем данные к формату пакеты из одного примера входных даных.
        # Это позволит выполнять дальнейшие вычисления единообразно.
        _input_2d = tf.reshape(tensor=input_2d, shape=(1, input_2d.shape[0], input_2d.shape[1], input_2d.shape[2]))
        _input_1d = tf.reshape(tensor=input_1d, shape=(1, input_1d.shape[0], input_1d.shape[1]))
    else:
        _input_2d = input_2d
        _input_1d = input_1d

    #  преобразуем 1d тензор в 2d тензор и конкатенируем тензоры по оси фильтра
    _input_2d = tf.concat(values=[_input_2d,
                                  tf.broadcast_to(tf.reshape(_input_1d, (batch_size, 1, 1, length * number_of_filters)),
                                                  [batch_size, height, weight, length * number_of_filters])],
                          axis=3)

    if not batch_mode:
        _input_2d = tf.reshape(tensor=_input_2d, shape=_input_2d.shape[1:])

    return _input_2d


class Model2D(Model):
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
        :param rate_of_pool_size_increase: Коэффициент увеличения окна объединения в каждом новом модуле модели
        по формуле P[i] = P[i-1] * ( 1 + rate_of_pool_size_increase)
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
        model: TensorFlowSequentialModel = TensorFlowSequentialModel()

        input_shape: tuple[int, int, int] = (self.input_2d_shape[0], self.input_2d_shape[1],
                                             self.input_2d_shape[2] + self.input_1d_shape[0] * self.input_1d_shape[1])
        model.add(tf.keras.Input(shape=input_shape))

        output_shape = tf.TensorShape(dims=input_shape)

        module_filters: float = self.start_filters
        module_kernel_size: list[float, float] = list(self.start_kernel_size)
        module_pool_size: list[float, float] = list(self.start_pool_size)

        # выполняем цикл пока не получим на выходе сверточной сети тензор с формой (1, 1, filters).
        # Пока не сведем размер 2D плоскости к точке с некоторым количеством фильтров.
        # Применять свертку и объединение к такой точке уже не требуется.
        while not (output_shape[0] == 1 and output_shape[1] == 1):
            # строим модуль, состоящий из последовательности слоев
            module_input_shape = output_shape  # форма входа совпадает с выходом последнего слоя предыдущего модуля

            # слой нормализации
            model.add(layer=BatchNormalization())

            # слой свертки
            conv: Conv2D = Conv2D(kernel_size=(int(module_kernel_size[0]), int(module_kernel_size[1])),
                                  input_shape=module_input_shape,
                                  filters=int(module_filters),
                                  padding="same",
                                  activation=tf.nn.softsign)
            model.add(layer=conv)
            # При расчете выходной формы тензора добавляем фиктивный BATCH SIZE, так как tensorflow ожидает его.
            output_shape = conv.compute_output_shape(input_shape=[1] + module_input_shape)[1:]

            # слой объединения
            max_pooling: MaxPooling2D = MaxPooling2D(pool_size=(int(module_pool_size[0]), int(module_pool_size[1])))
            model.add(layer=max_pooling)
            # При расчете выходной формы тензора добавляем фиктивный BATCH SIZE, так как tensorflow ожидает его.
            output_shape = max_pooling.compute_output_shape(input_shape=[1] + output_shape)[1:]

            module_filters *= (1 + self.rate_of_filters_increase)
            module_kernel_size[0] *= (1 + self.rate_of_kernel_size_increase)
            module_kernel_size[1] *= (1 + self.rate_of_kernel_size_increase)
            module_pool_size[0] *= (1 + self.rate_of_pool_size_increase)
            module_pool_size[1] *= (1 + self.rate_of_pool_size_increase)

        # выход модели окончательно вычисляет полносвязный слой персептрона с линейной функцией активации нейронов
        model.add(layer=Flatten())
        model.add(layer=BatchNormalization())
        model.add(layer=Dense(self.output_length, activation=None))

        return model

    def predict(self, model_input: Tensor, training: bool) -> Tensor:
        """
        Вычисление функции модели.
        :param model_input: Тензор входных данных. Может иметь форму (HEIGHT, WEIGHT, FILTERS) или
        (BATCH SIZE, HEIGHT, WEIGHT, FILTERS).
        :param training: Признак режима обучения модели.
        :return: Тензор результата вычислений в форме (OUTPUTS) или (BATCH SIZE, OUTPUTS),
        в зависимости от формы входа.
        """

        batch_mode: bool = model_input.shape.ndims == 4

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
