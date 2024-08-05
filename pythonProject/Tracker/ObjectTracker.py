import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Этот класс позволяет трекать любые объекты по нажатию мыши. От танокв до обычных кусов
class VideoTracker:
    def __init__(self, video_path, win_name="Tracking by contours"):
        self.alive = True  # Для цикла while, чтоб обрабатывать видео пока оно не закончится, либо пока не нажмется буква 'q' для выхода
        self.mode = 1  # Режимы от 1 до 6. Для переключения в разные режимы для наглядной демострации каждого этапа от обработки нужной области до самого начала трекинга
        self.win_name = win_name  # Название окна куда выводить каждый новый обработанный фрейм
        self.count_no_found_contours = 0  # Счетчик, который дает возможность в цикле 5 попыток на нахождения контура
        self.is_paused = False  # Переменная, которая ставит видео на паузу для наглядной демострации каждого этапа
        self.radius_rows = 50  # Высота от центральной точки куда клацнуть мышкой для выбора области за которой хотим следить. Соответственно общая высота области будет 100px
        self.radius_cols = 70  # Ширина от центральной точки куда клацнуть мышкой для выбора области за которой хотим следить. Соответственно общая ширина области будет 100px
        self.tracker_type = "CSRT"  # Выбранный алгоритм трекинга. Channel and Spatial Reliability. Для меня он показался самый надежный и достаточно быстрый. Пробовал dlib и другие алгоритмы - все не пронравилось, все имеют свои минусы
        self.is_tracker_initialized = False  # Это для цикла, чтоб понимать что трекинг уже начался, область выбрана и можно запускать и апдейтить трекинг
        self.tracker, self.bbox = None, None  # Сам объект трекинга и области которую трекать
        self.selected_point = None  # Точка которая ставится мышкой
        self.source = cv2.VideoCapture(video_path)  # Читаем видео файл
        self.copy_from_source = None  # Копия фрейма из self.source
        self.colors = {
            "blue": (255, 0, 0),
            "green": (0, 255, 0),
            "red": (0, 0, 255)
        }
        self.line_thickness = 2
        self.contour_thickness = 1

        # self.model = load_model('C:\\Users\\Alex\\Downloads\\unetSegmentation.keras')
        # Файлик не коммичу на гитхаб Силшком много весит. Этот файлик можно получить тут
        # https://drive.google.com/file/d/1X5lq5kUzBdhq_ntEud91zUFLd_w8qWLV/view?usp=sharing
        self.model = load_model('NeuralNetwork/TrainResults/unetSegmentation.keras')

        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)  # ДАем название окну куда выводить видео и инициализируем это окно
        cv2.setMouseCallback(win_name, self.mouse_click_event)  # Подписывамемся на эвент клацанья мышки

    def draw_rectangle(self, frame, start_point, end_point, color, thickness):
        cv2.rectangle(frame, start_point, end_point, color, thickness)

    # Сам евент на клацанье мышкой. Получение координат точки куда клацнули.
    def mouse_click_event(self, event, col, row, flags, param):
        #  Левовой мышкой получаем точку и инициализируем трекинг.
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selected_point = None
            self.is_tracker_initialized = False

            print("Init CSRT tracker")
            # Подбирал параметры которые лучше всего работают для моей задачи
            params = cv2.TrackerCSRT_Params()
            params.use_hog = True
            params.use_color_names = False
            params.use_gray = False
            params.use_rgb = True
            params.use_channel_weights = True
            params.use_segmentation = True
            self.tracker = cv2.TrackerCSRT_create(params)  # Создаем трекер

            self.selected_point = (col, row)  # Получаем координаты клацнутой точки

        # Сбрасываем все настройки, перестаем трекать
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.selected_point = None
            self.is_tracker_initialized = False
            if self.is_paused:  # В режиме паузы выводим чистое окно без всех нарисованых ректанглов
                cv2.imshow(self.win_name, self.copy_from_source)

    # Получаем координаты начала и конца области которую будем поддавать обработке
    def get_area_points(self, frame, x, y):
        x1 = max(x - self.radius_rows, 0)
        y1 = max(y - self.radius_cols, 0)
        x2 = min(x + self.radius_rows, frame.shape[0])
        y2 = min(y + self.radius_cols, frame.shape[1])

        return x1, y1, x2, y2

    # Наложения фильтра Собеля на фрейм. По сути делаем тоже самое что делали на лекции. Размываем картинку, переводим в черно-белую, считаем magnitude
    def sobol_filter_set(self, frame):
        sobel_frame = cv2.medianBlur(frame, ksize=3)

        frame_gray = cv2.cvtColor(sobel_frame, cv2.COLOR_BGR2GRAY)
        gray = frame_gray / 255
        grad_x = cv2.Sobel(gray, ddepth=-1, dx=1, dy=0)
        grad_y = cv2.Sobel(gray, ddepth=-1, dx=0, dy=1)

        # Compute magnitude
        sobel = np.sqrt(grad_x ** 2 + grad_y ** 2)
        sobel = sobel / np.max(sobel)

        sobel = (sobel * 255).astype(np.uint8)

        # Преобразуйте одноканальное изображение Собеля в трехканальное изображение. Нужно дальше
        sobel_3c = cv2.merge([sobel, sobel, sobel])
        return sobel, sobel_3c

    # Находим лучший трешхолд методом OTSU. Пробовал и адаптив трешхолд и метод саувола. Но как по мне для моей задачи, с ограниченного областью, этот метод работает лучше всего.
    def find_the_best_threshold(self, frame):
        histo_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        opt_THRESHOLD, _ = cv2.threshold(histo_img, 0, 255, cv2.THRESH_OTSU)

        return opt_THRESHOLD

    # Находим контура в выбранной области thresholded_frame и уже наложенным подобранным трешхолдом
    def find_contours(self, thresholded_frame, original_frame,
                      is_draw_contours=False,
                      is_draw_rectangle_around_contours=False):
        x11, y11, w11, h11 = 0, 0, 0, 0
        contours, hierarchy = cv2.findContours(thresholded_frame, cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)
        selected_contour_index = -1
        isCountorFound = False
        # Проверяем попадает ли наша точка selected_point в какую-то из найденных котуров с погрешностью в 3 пикселя. Если попадает, то считаем что нашли контур
        for i, contour in enumerate(contours):
            measure = cv2.pointPolygonTest(contour, self.selected_point, True)
            if measure >= -3:
                selected_contour_index = i  # Если нашли контур то присваиваем переменной индекс найденного контура и выходим из цикла
                break
        if not selected_contour_index == -1:
            contour = contours[selected_contour_index]
            x11, y11, w11, h11 = cv2.boundingRect(contour) # Получаем прямоугольник вокрцуг найденного контура
            # Костыль) Если высота или ширина области меньше 15 пикселей, то расширяем ее немного, при этом смещаем начальную точку прямоугольника равномерно на сколько была расширина область
            if w11 < 15:
                x11 = int(x11 - (15 - w11) / 2)
                w11 = 20
            if h11 < 15:
                y11 = int(y11 - (15 - h11) / 2)
                h11 = 15
            if is_draw_rectangle_around_contours:
                self.draw_rectangle(original_frame, (x11, y11), (x11 + w11, y11 + h11), self.colors["blue"], self.line_thickness)
                # cv2.rectangle(original_frame, (x11, y11), (x11 + w11, y11 + h11), self.colors["blue"], 2)
            isCountorFound = True

        if is_draw_contours:
            # Отображаем найденные контуры, если так хотим
            cv2.drawContours(original_frame, contours, -1, self.colors["green"], self.contour_thickness,
                             cv2.LINE_AA)

        return isCountorFound, x11, y11, x11 + w11, y11 + h11

    # Это защита, если вдруг не нашлось контуров - точка не принадлежит ни одному контуру. Тогда поподаем в этот метод.
    # Он считает количество белых пикселей вокруг нашей точки. Там где их максимальное количество ту область и выделяет.
    # Область поиска делю по гридам на 10х10. Поиски начинаю от центральной точки. Если нужно объединяю гриды
    # Минус этого метода, что он всегда вернет значения, даже если не найдет белые пиксели.
    # Чтоб понять почему, обратите внимание на сторочки где min_x, min_y = width, height - Инициализируем переменные для хранения объединенной области. Потом смотрим дальеш
    def find_max_points_in_current_area(self, thresholded_frame, grid_size=10, threshold_ratio=0.8, is_neural_network_segmantation=False):
        # selectedRegionSettedThreshold = cv2.adaptiveThreshold(sobel, 255,
        #                                                       cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 20)
        # grid_size = 20
        # threshold_ratio = 0.7  # Порог для объединения ячеек (80% от максимального). Для понимания нужно ли объединять гриды

        # Получаем размеры изображения
        height, width = thresholded_frame.shape
        # Определяем центральную ячейку
        center_x, center_y = width // 2, height // 2
        center_cell_x = center_x // grid_size
        center_cell_y = center_y // grid_size
        # Создаем карту для хранения количества белых пикселей в каждой ячейке
        points_map = np.zeros((height // grid_size, width // grid_size), dtype=int)

        # Проходимся по изображению с шагом равным размеру сетки
        for y in range(0, height, grid_size):
            for x in range(0, width, grid_size):
                # Получаем текущую ячейку
                cell = thresholded_frame[y:y + grid_size, x:x + grid_size]

                # Считаем количество белых пикселей в ячейке
                num_points = cv2.countNonZero(cell)

                # Сохраняем количество белых пикселей в карте
                points_map[y // grid_size, x // grid_size] = num_points

        # Находим количество белых пикселей в центральной ячейке
        center_points = points_map[center_cell_y, center_cell_x]

        # Защита. Если нейронка не нашла сегмент объекта, то возвращаем нули и даем возможность найти контурам.
        # Если же и контура не нашли, то снова попадаем в этот метод, только уже с другим фреймом на который наложены разные фильтра
        # дальше в любом случае получим какую-то область, даже если белых пикселей не найдем
        if is_neural_network_segmantation:
            zeros_map = np.zeros((height // grid_size, width // grid_size), dtype=int)
            if np.array_equal(points_map, zeros_map):
                return 0, 0, 0, 0

        # Определяем порог для объединения ячеек
        merge_threshold = center_points * threshold_ratio

        # Находим ячейки, которые нужно объединить
        regions_to_merge = np.argwhere(points_map >= merge_threshold)

        # Инициализируем переменные для хранения объединенной области
        min_x, min_y = width, height
        max_x, max_y = 0, 0

        # Проходимся по ячейкам, которые нужно объединить
        for region in regions_to_merge:
            ry, rx = region
            cell_x, cell_y = rx * grid_size, ry * grid_size

            # Проверяем, что ячейка не находится по краям изображения
            if (abs(rx - center_cell_x) > 1 or abs(ry - center_cell_y) > 1):
                continue

            min_x = min(min_x, cell_x)
            min_y = min(min_y, cell_y)
            max_x = max(max_x, cell_x + grid_size)
            max_y = max(max_y, cell_y + grid_size)

        return min_x, min_y, max_x, max_y

    # Тут стартует трекер. Передаем в него весь фрейм и коодинаты выбранной области найденной по контурам или максБелых пикселей
    def start_tracker(self, frame, x1, y1, x2, y2):
        x11 = min(x1, x2)
        y11 = min(y1, y2)
        x12 = max(x1, x2)
        y12 = max(y1, y2)
        width = x12 - x11
        height = y12 - y11

        bbox = (x11, y11, width, height)

        tracking = self.tracker.init(frame, bbox)
        self.is_tracker_initialized = True
        # Сбрасываем координыты выбранной области
        self.selected_region = None

        return tracking, bbox

    # Обработка фрейма. В зависимости от того стартанул трекер или нет, выбрана область или нет попоадает в соответсвующее условие
    def process_frame(self, frame):
        if self.is_tracker_initialized:
            self.track_object(frame)
        elif self.selected_point:
            self.process_selected_region(frame)
        return frame

    # Обновления трекера и перерисовываем прямоугольник
    def track_object(self, frame):
        tracking, bbox = self.tracker.update(frame)
        x11, y11, width, height = bbox
        cv2.rectangle(frame, (int(x11), int(y11)), (int(x11 + width), int(y11 + height)), self.colors["blue"], 2)

    # Метод вызывается, когда поставлена точка мышью. Тут и вызываются потом методы наложения фильтров и поиск контуров
    def process_selected_region(self, frame):
        y, x = self.selected_point
        cv2.rectangle(frame, (y - self.radius_cols, x - self.radius_rows), (y + self.radius_cols, x + self.radius_rows),
                      self.colors["red"], 2)
        x1, y1, x2, y2 = self.get_area_points(frame, x, y)

        # Сначала проверяем нейронку с сегментацией, нашла ли она что-то. Если она найдет, то начнет трекинг
        frame[x1:x2, y1:y2, :], is_found_segments = self.find_segments_and_draw_rectangle(frame[x1:x2, y1:y2, :],
                                                                                          frame,
                                                                                          x1, y1,
                                                                                          is_view_mask=False,
                                                                                          is_draw_rectangle=True)
        # если нейронка не нашла ничего, то накладываем фильтры и ищем контуры.
        # Тут в любом случае что-то надем, пусть даже не правильно и начнем трекинг
        if not is_found_segments:
            sobel, sobel_3c = self.sobol_filter_set(frame[x1:x2, y1:y2, :])
            opt_threshold = self.find_the_best_threshold(sobel_3c)
            ret, selected_region_setted_threshold = cv2.threshold(sobel, opt_threshold, 255, cv2.THRESH_BINARY)
            self.find_and_draw_contours(selected_region_setted_threshold,
                                        frame, x1, x2, y1, y2)

    # Метод вызывает метод поиска контуров. Если не надены, то вызывает метод поиска максимального сосредоточения белых пикселей. После всего стартует трекер
    # Даем 5 попыток найти контуры. Идея в том что обрабатывается каждый раз новый следующий кадр, соответсвенно точка смещается и может попасть в контур
    def find_and_draw_contours(self, thresholded_frame, original_frame, x1, x2, y1, y2,
                               is_draw_contours=False, is_draw_rectangle_around_contours=True):
        is_contour_found, x11, y11, x12, y12 = self.find_contours(thresholded_frame, original_frame[x1:x2, y1:y2, :],
                                                                  is_draw_contours, is_draw_rectangle_around_contours)
        if not is_contour_found:
            self.count_no_found_contours += 1
            if self.count_no_found_contours >= 5:
                x11, y11, x12, y12 = self.find_max_points_in_current_area(thresholded_frame)
        elif (x12 < 15 or y12 < 15) and is_contour_found:
            x11, y11, x12, y12 = self.find_max_points_in_current_area(thresholded_frame)
            self.count_no_found_contours = 0
            # Даем возможность 5 попыток на нахождения контуров объекта.
            # Если найдет то count_no_found_contours == 0 и начнет трекать.
            # Если не найдет контуры более 5 раз, то начнем трекать по максимальному количеству точек начиная от центра
        if self.count_no_found_contours == 0 or self.count_no_found_contours >= 5:
            if not self.is_paused:
                self.start_tracker(original_frame, x11 + y1, y11 + x1, x12 + y1, y12 + x1)
            elif self.is_paused and is_draw_rectangle_around_contours:  # Это для наглядности что найдено в режиме паузы
                self.draw_rectangle(original_frame[x1:x2, y1:y2, :], (x12, y12), (x11, y11), self.colors["blue"], self.line_thickness)

    def find_segments_and_draw_rectangle(self, area_frame, original_frame,
                                         x1, y1,
                                         is_view_mask=False, is_draw_rectangle=True):
        rows, cols, _ = area_frame.shape
        frame_rgb = cv2.cvtColor(area_frame, cv2.COLOR_BGR2RGB)
        resized_frame = cv2.resize(frame_rgb, (128, 128))
        frame_array = img_to_array(resized_frame) / 255.0  # Нормализация изображения
        # Добавление размерности batch, так как model.predict ожидает batch
        frame_array = np.expand_dims(frame_array, axis=0)
        # Предсказание
        frame_pred = self.model.predict(frame_array)
        # Обработка предсказания для создания маски
        frame_mask = np.argmax(frame_pred[0], axis=-1)
        frame_mask = np.expand_dims(frame_mask, axis=-1)

        # Изменение размера маски для соответствия размеру оригинального изображения
        frame_mask_resized = cv2.resize(frame_mask.astype(np.uint8), (cols, rows))

        # Умножаем на 255, чтоб видеть польностью белый объект
        frame_mask_resized = (frame_mask_resized * 255).astype(np.uint8)

        if is_view_mask:

            # Преобразуйте одноканальное изображение Собеля в трехканальное изображение.
            frame_mask_resized_3d = cv2.merge([frame_mask_resized, frame_mask_resized, frame_mask_resized])
            # отображаем в выбранной области данную маску
            area_frame = frame_mask_resized_3d
        is_found_segments = True
        if is_draw_rectangle:
            # Нейронка выдаст нам маску и можно теперь считать количество белых пикселей в выбранной области после чего рисуем наш прямоугольник
            # Методом тыка определил, что параметры grid_size=20, threshold_ratio=0.7 лучше объединяют нужные ячейки при работе с сегментацией
            x11, y11, x12, y12 = self.find_max_points_in_current_area(frame_mask_resized,
                                                                      grid_size=20,
                                                                      threshold_ratio=0.7,
                                                                      is_neural_network_segmantation=True)
            # Определяем нашли ли мы что-то
            if x11 == 0 and y11 == 0 and x12 == 0 and y12 == 0:
                is_found_segments = False

            self.draw_rectangle(area_frame, (x12, y12), (x11, y11), self.colors["blue"],
                                self.line_thickness)

            # Если нашли, то начинаем трекинг
            if not self.is_paused and is_found_segments:
                self.start_tracker(original_frame, x11 + y1, y11 + x1, x12 + y1, y12 + x1)
        return area_frame, is_found_segments

    # Сам цикл, который обрабатывает каждый новый фрейм видео.
    # Еще обрабатывает нажатие клавиш клавиатуры, вторыми выставляются режимы для наглядности прохождения каждого этапа
    def run(self):
        while self.alive:
            if not self.is_paused:
                has_frame, frame = self.source.read()
                if not has_frame:
                    break
                self.copy_from_source = frame.copy()
                frame = self.process_frame(frame)
                cv2.imshow(self.win_name, frame)
            else:
                self.process_paused_state()

            key = cv2.waitKey(50 if self.is_paused else 24)
            self.handle_key_press(key)

        self.source.release()
        cv2.destroyWindow(self.win_name)

    # В режиме паузы обрабатывает нажатие клавишь клавиатуры (1,2,3,4,5,6) для выставления разных модов для наглядности прохождения каждого этапа
    def process_paused_state(self):
        if self.selected_point:
            # Работаем с копией фрейма чтоб не загрязнять исходную картинку
            frame_copy = self.copy_from_source.copy()
            (col, row) = self.selected_point
            # Рисуем область которую будем поддавать обработке фильтров и поиску контуров
            self.draw_rectangle(frame_copy, (col - self.radius_cols, row - self.radius_rows),
                                (col + self.radius_cols, row + self.radius_rows), self.colors["red"],
                                self.line_thickness)
            if self.mode in [2, 3, 4, 5, 6]:
                # Получаем координаты этой области
                x1, y1, x2, y2 = self.get_area_points(frame_copy, row, col)
                # Накладываем фильтр Собеля
                sobel, sobel_3c = self.sobol_filter_set(frame_copy[x1:x2, y1:y2, :])
                if self.mode in [2, 3]:
                    # В режиме 2 и 3 отображаем в выбранной области результат наложения фильтра Собеля
                    frame_copy[x1:x2, y1:y2, :] = sobel_3c
                # Находим лучший трешхолд
                opt_threshold = self.find_the_best_threshold(sobel_3c)
                # Накладываем найденный трешхолд
                _, selected_region_setted_threshold = cv2.threshold(sobel, opt_threshold, 255, cv2.THRESH_BINARY)
                if self.mode in [3, 4]:
                    # В режиме 3 и 4 отображаем найденные контура
                    self.find_and_draw_contours(selected_region_setted_threshold, frame_copy, x1, x2, y1, y2,
                                                is_draw_contours=True, is_draw_rectangle_around_contours=False)
                if self.mode == 5:
                    # Тут отображаем контура и рисуем прямоугольник вокруг найденого контура
                    self.find_and_draw_contours(selected_region_setted_threshold, frame_copy, x1, x2, y1, y2,
                                                is_draw_contours=True, is_draw_rectangle_around_contours=True)
                if self.mode == 6:
                    # Тут выключаем отображения контуров и отображаем лишь прямоугольник вокруг найденного контура
                    self.find_and_draw_contours(selected_region_setted_threshold, frame_copy, x1, x2, y1, y2,
                                                is_draw_contours=False, is_draw_rectangle_around_contours=True)
            if self.mode in [7, 8, 9]:
                # Получаем координаты этой области
                x1, y1, x2, y2 = self.get_area_points(frame_copy, row, col)
                if self.mode == 7:
                    # Тут применяем сегментацию обученной нейронки и показываем что она нашла.
                    frame_copy[x1:x2, y1:y2, :], _ = self.find_segments_and_draw_rectangle(frame_copy[x1:x2, y1:y2, :],
                                                                                           frame_copy,
                                                                                           x1, y1,
                                                                                           is_view_mask=True,
                                                                                           is_draw_rectangle=False)
                if self.mode == 8:
                    # Тут тоже самое ,что и в 7, только еще рисуем прямоугольник вокруг найденой области.
                    frame_copy[x1:x2, y1:y2, :], _ = self.find_segments_and_draw_rectangle(frame_copy[x1:x2, y1:y2, :],
                                                                                           frame_copy,
                                                                                           x1, y1,
                                                                                           is_view_mask=True,
                                                                                           is_draw_rectangle=True)
                if self.mode == 9:
                    # Показываем результат уже на читой картинке
                    frame_copy[x1:x2, y1:y2, :], _ = self.find_segments_and_draw_rectangle(frame_copy[x1:x2, y1:y2, :],
                                                                                           frame_copy,
                                                                                           x1, y1,
                                                                                           is_view_mask=False,
                                                                                           is_draw_rectangle=True)

            cv2.imshow(self.win_name, frame_copy)

    # Обработка клавишь клавиатуры и выставление нужных режимов
    def handle_key_press(self, key):
        if key == ord('q'):
            self.alive = False
        elif key == ord('p'):
            self.mode = 1
            self.is_paused = not self.is_paused
        elif key in map(ord, ["1", "2", "3", "4", "5", "6", "7", "8", "9"]):
            self.mode = key - ord("0")


if __name__ == "__main__":
    # video_path = "C:\\AllFromVladPC\\learning\\OpenCv\\Section_01\\Tank3\\1.mp4"
    video_path = "/home/user/Downloads/Telegram Desktop/1.mp4"
    tracker = VideoTracker(video_path)
    tracker.run()