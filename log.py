import sys
from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox, QInputDialog, QListWidgetItem
from PyQt6.QtGui import QPixmap, QPainter, QPen, QIcon, QColor, QBrush
from PyQt6.QtCore import Qt
from design import GraphicEditor
from PIL import Image, ImageOps
from PIL.ImageQt import ImageQt
import matplotlib.pyplot as plt
from io import BytesIO
import matplotlib.font_manager as fm
import numpy as np
from scipy import ndimage
from PyQt6.QtWebEngineWidgets import QWebEngineView
import plotly.graph_objs as go
import plotly.io as pio


class EditorLogic(GraphicEditor):
    def __init__(self):
        super().__init__()
        self.image = None
        self.original_image = None
        self.history = []
        self.redo_stack = []
        self.temp_image = None
        self.current_gamma = 1.0
        self.scale_factor = 1.0  # 100% масштаб по умолчанию
        self.has_new_history_items = False  # Флаг для отслеживания новых действий в истории
        self.history_action_text = "История действий"

        actions = self.get_actions()
        actions['open'].triggered.connect(self.open_file)
        actions['save'].triggered.connect(self.save_file)
        actions['rotate_left'].triggered.connect(self.rotate_left)
        actions['rotate_right'].triggered.connect(self.rotate_right)
        actions['flip_horizontal'].triggered.connect(lambda: self.modify_image(self.flip_horizontal))
        actions['flip_vertical'].triggered.connect(lambda: self.modify_image(self.flip_vertical))
        actions['grayscale'].triggered.connect(lambda: self.modify_image(self.apply_grayscale))
        actions['invert'].triggered.connect(lambda: self.modify_image(self.apply_invert))
        actions['binary'].triggered.connect(lambda: self.modify_image(self.apply_binary))
        actions['content_aware'].triggered.connect(lambda: self.modify_image(self.adjust_image_scale))
        actions['undo'].triggered.connect(self.undo_action)
        actions['redo'].triggered.connect(self.redo_action)
        actions['reference'].triggered.connect(self.reference_action)
        actions['gamma_correction'].triggered.connect(lambda: self.modify_image(self.apply_gamma_correction))
        # actions['resize_custom'].triggered.connect(self.resize_image_dialog)
        actions['crop_square'].triggered.connect(lambda: self.crop_template('square'))
        actions['crop_4_3'].triggered.connect(lambda: self.crop_template('4:3'))
        actions['crop_16_9'].triggered.connect(lambda: self.crop_template('16:9'))
        # actions['restore_original'].triggered.connect(self.restore_original)
        actions['content_aware'].triggered.connect(self.content_aware_resize_dialog)

        # действие для отображения/скрытия истории
        actions['history'].triggered.connect(self.toggle_history_panel)

        # действие для переключения темы
        actions['toggle_theme'].triggered.connect(self.toggle_theme)

        # действий масштабирования
        actions['zoom_in'].triggered.connect(self.zoom_in)
        actions['zoom_out'].triggered.connect(self.zoom_out)
        actions['zoom_original'].triggered.connect(self.zoom_original)

        self.undo_button.clicked.connect(self.undo_action)
        self.redo_button.clicked.connect(self.redo_action)

        # кнопка закрытия истории
        self.close_history_button.clicked.connect(self.hide_history_panel)

        self.gamma_slider.valueChanged.connect(self.preview_gamma_correction)
        self.reset_gamma_button.clicked.connect(self.reset_gamma)
        self.apply_gamma_button.clicked.connect(self.apply_gamma)

        self.gamma_group.setEnabled(False)

        # включение обработки колеса мыши для масштабирования
        self.scroll_area.wheelEvent = self.wheel_event

        self.update_statusbar()

        # применяем текущую тему
        self.apply_theme()

    def wheel_event(self, event):
        if self.image and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y()
            if delta > 0:
                self.zoom_in()
            else:
                self.zoom_out()
        else:
            super(self.scroll_area.__class__, self.scroll_area).wheelEvent(event)

    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Открыть файл", "", "Изображения (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            try:
                self.image = Image.open(file_path)
                self.original_image = self.image.copy()
                self.history.clear()
                self.redo_stack.clear()
                self.history.append(self.image.copy())
                self.history_list.addItem(f"Открыт файл: {file_path.split('/')[-1]}")
                self.scale_factor = 1.0  # Сброс масштаба до 100%
                self.display_image()
                self.display_histogram()
                self.gamma_group.setEnabled(True)
            except Exception as e:
                print(f"Ошибка загрузки файла: {e}")

    def save_file(self):
        if not self.image:
            print("Нет изображения для сохранения.")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Сохранить файл", "", "Изображения (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            try:
                self.image.save(file_path)
                self.history_list.addItem(f"Файл сохранён: {file_path.split('/')[-1]}")
                print(f"Файл сохранён: {file_path}")
            except Exception as e:
                print(f"Ошибка сохранения файла: {e}")

    def reference_action(self):
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("О программе")
        msg_box.setText("Графический редактор\nВерсия: 1.0\nАвтор: Штырхунова Анастасия")
        msg_box.setIcon(QMessageBox.Icon.Information)
        msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg_box.exec()

###########################################################################################################

    def display_histogram(self):
        if not self.image:
            print("Нет изображения для построения гистограммы.")
            return

        from matplotlib.ticker import MaxNLocator, FuncFormatter

        grayscale_image = self.image.convert("L")
        histogram = grayscale_image.histogram()
        font_path = "fonts/Roboto-Regular.ttf"
        roboto_font = fm.FontProperties(fname=font_path)

        # получаем актуальный размер QLabel
        label_width = self.histogram_label.width() if self.histogram_label.width() > 0 else 400
        label_height = self.histogram_label.height() if self.histogram_label.height() > 0 else 180
        figsize = (label_width / 96, label_height / 96)
        dpi = 96

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # устанавливаем цвета в зависимости от темы
        if self.current_theme == "dark":
            bg_color = '#232323'
            text_color = 'white'
            bar_color = '#D0D0D0'
            grid_color = '#444444'
            spine_color = '#AAAAAA'
        else:
            bg_color = '#F5F5F5'
            text_color = '#333333'
            bar_color = '#666666'
            grid_color = '#CCCCCC'
            spine_color = '#999999'

        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)
        ax.bar(range(256), histogram, color=bar_color, width=1.0, edgecolor='#555555', linewidth=0.5)

        ax.set_title("", fontsize=0)
        ax.set_xlabel("Яркость", fontproperties=roboto_font, fontsize=13, color=text_color, labelpad=8)
        ax.set_ylabel("Количество", fontproperties=roboto_font, fontsize=13, color=text_color, labelpad=8)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True, prune='lower'))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: (
            f"{int(x / 1_000_000)}M" if x >= 1_000_000 else
            f"{int(x / 1_000)}k" if x >= 1_000 else
            f"{int(x)}"
        )))
        ax.set_xticks(range(0, 256, 64))
        ax.tick_params(axis='x', labelsize=10, colors=text_color)
        ax.tick_params(axis='y', labelsize=10, colors=text_color)
        ax.grid(color=grid_color, linestyle='--', linewidth=0.5, alpha=0.7)
        for spine in ax.spines.values():
            spine.set_color(spine_color)
            spine.set_linewidth(1)
        plt.tight_layout(pad=1.5)

        buffer = BytesIO()
        plt.savefig(buffer, format="PNG", facecolor=bg_color, dpi=dpi, bbox_inches="tight")
        buffer.seek(0)
        plt.close()
        histogram_image = Image.open(buffer)
        qt_image = ImageQt(histogram_image)
        pixmap = QPixmap.fromImage(qt_image)
        self.histogram_label.setPixmap(pixmap)
        self.histogram_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        buffer.close()
        print("Гистограмма обновлена.")

    ###########################################################################################################

    def crop_template(self, template):
        """Кадрирование по шаблону: 'square', '4:3', '16:9'"""
        if not self.image:
            return

        width, height = self.image.size

        if template == 'square':
            # квадрат - используем меньшую из сторон, crop по центру
            size = min(width, height)
            left = (width - size) // 2
            top = (height - size) // 2
            right = left + size
            bottom = top + size

        elif template == '4:3':
            # соотношение 4:3
            target_ratio = 4 / 3
            if width / height > target_ratio:
                new_width = int(height * target_ratio)
                left = (width - new_width) // 2
                right = left + new_width
                top = 0
                bottom = height
            else:
                new_height = int(width / target_ratio)
                top = (height - new_height) // 2
                bottom = top + new_height
                left = 0
                right = width

        elif template == '16:9':
            # соотношение 16:9
            target_ratio = 16 / 9
            if width / height > target_ratio:
                new_width = int(height * target_ratio)
                left = (width - new_width) // 2
                right = left + new_width
                top = 0
                bottom = height
            else:
                new_height = int(width / target_ratio)
                top = (height - new_height) // 2
                bottom = top + new_height
                left = 0
                right = width

        else:
            return

        # проверка границ
        left = max(0, left)
        top = max(0, top)
        right = min(width, right)
        bottom = min(height, bottom)
        if right <= left or bottom <= top:
            return

        self.history.append(self.image.copy())
        self.redo_stack.clear()
        self.image = self.image.crop((left, top, right, bottom))
        self.display_image()
        self.display_histogram()
        self.history_list.addItem(f"Кадрирование: {template}")

    def crop_image(self, left, top, right, bottom):
        if self.image:
            self.image = self.image.crop((left, top, right, bottom))

###########################################################################################################

    def preview_gamma_correction(self):
        """Превью гамма-коррекции при перемещении слайдера"""
        if not self.image:
            return

        # Если это первое изменение, сохраняем исходное изображение для возможности сброса
        if not self.temp_image:
            self.temp_image = self.image.copy()

        # Получаем значение гаммы из слайдера (делим на 100, т.к. QSlider работает с целыми)
        gamma = self.gamma_slider.value() / 100
        self.current_gamma = gamma
        self.gamma_label.setText(f"γ = {gamma:.2f}")

        # Применяем гамма-коррекцию к временному изображению
        img_array = np.array(self.temp_image).astype(np.float32) / 255.0
        corrected = np.power(img_array, 1.0 / gamma)
        corrected = (corrected * 255).clip(0, 255).astype(np.uint8)
        self.image = Image.fromarray(corrected)

        # Обновляем отображение без добавления в историю
        self.display_image()
        self.display_histogram()

###########################################################################################################
# не работает

    def content_aware_resize_dialog(self):
        """Диалог для масштабирования с учетом содержимого"""
        if not self.image:
            return

        # Запрашиваем новые размеры
        width, ok_w = QInputDialog.getInt(self, "Масштабирование с учетом содержимого",
                                           "Введите новую ширину:", value=self.image.width)
        if not ok_w:
            return

        height, ok_h = QInputDialog.getInt(self, "Масштабирование с учетом содержимого",
                                            "Введите новую высоту:", value=self.image.height)
        if not ok_h:
            return

        self.history.append(self.image.copy())
        self.redo_stack.clear()

        self.image = self.content_aware_resize(self.image, width, height)
        self.display_image()
        self.display_histogram()
        self.history_list.addItem(f"Масштабирование с учетом содержимого: {width}x{height}")

    def content_aware_resize(self, img, target_width, target_height):
        """Упрощенная реализация алгоритма Seam Carving для масштабирования с учетом содержимого"""
        # Конвертируем PIL Image в numpy array
        img_array = np.array(img)
        
        # Получаем текущие размеры
        current_height, current_width = img_array.shape[:2]
        
        # Определяем, насколько нужно изменить размеры
        width_diff = target_width - current_width
        height_diff = target_height - current_height
        
        # Сначала изменяем ширину
        result_image = img_array
        
        # Уменьшаем ширину
        if width_diff < 0:
            result_image = self.reduce_width(result_image, abs(width_diff))
        # Увеличиваем ширину
        elif width_diff > 0:
            result_image = self.increase_width(result_image, width_diff)
        
        # Затем изменяем высоту
        # Уменьшаем высоту
        if height_diff < 0:
            # Поворачиваем изображение, чтобы работать со строками как со столбцами
            rotated = np.rot90(result_image, k=1)
            # Уменьшаем ширину повернутого изображения
            rotated = self.reduce_width(rotated, abs(height_diff))
            # Возвращаем изображение в исходную ориентацию
            result_image = np.rot90(rotated, k=3)
        # Увеличиваем высоту
        elif height_diff > 0:
            # Поворачиваем изображение
            rotated = np.rot90(result_image, k=1)
            # Увеличиваем ширину повернутого изображения
            rotated = self.increase_width(rotated, height_diff)
            # Возвращаем изображение в исходную ориентацию
            result_image = np.rot90(rotated, k=3)
        
        # Конвертируем обратно в PIL Image
        return Image.fromarray(result_image.astype(np.uint8))

    def reduce_width(self, image, num_pixels):
        """Уменьшает ширину изображения, удаляя швы с минимальной энергией"""
        result = np.copy(image)
        for i in range(num_pixels):
            # Вычисляем энергию
            energy = self.calc_energy_map(result)
            # Вычисляем кумулятивную энергию
            cumulative = self.calc_cumulative_energy(energy)
            # Находим шов с минимальной энергией
            seam = self.find_min_seam(cumulative)
            # Удаляем шов
            result = self.remove_seam(result, seam)
        return result
    
    def increase_width(self, image, num_pixels):
        """Увеличивает ширину изображения, добавляя швы с плавной интерполяцией"""
        # Создаем копию исходного изображения
        result = np.copy(image)
        
        # Добавляем швы по одному, пересчитывая карту энергии после каждого добавления
        for i in range(num_pixels):
            # Вычисляем энергию
            energy = self.calc_energy_map(result)
            # Вычисляем кумулятивную энергию
            cumulative = self.calc_cumulative_energy(energy)
            # Находим шов с минимальной энергией
            seam = self.find_min_seam(cumulative)
            # Добавляем шов с плавной интерполяцией
            result = self.duplicate_seam(result, seam)
            
            # Каждые 20 пикселей применяем сглаживание для устранения артефактов
            # Уменьшаем частоту применения фильтра для сохранения яркости
            if (i + 1) % 20 == 0 and i > 0:
                result = self.apply_smooth_filter(result)
        
        return result
    
    def calc_energy_map(self, image):
        """Вычисляет карту энергии изображения с помощью оператора Собеля"""
        # Преобразуем в оттенки серого для упрощения расчетов
        gray = np.mean(image, axis=2).astype(np.float64)
        
        # Вычисляем градиенты с помощью оператора Собеля
        energy_x = np.absolute(ndimage.sobel(gray, axis=1))
        energy_y = np.absolute(ndimage.sobel(gray, axis=0))
        
        # Суммируем энергию
        return energy_x + energy_y
    
    def calc_cumulative_energy(self, energy):
        """Вычисляет кумулятивную карту энергии"""
        height, width = energy.shape
        cumulative = np.copy(energy)
        
        # Заполняем кумулятивную карту энергии
        for i in range(1, height):
            for j in range(width):
                # Находим минимальную энергию из трех возможных путей
                if j == 0:  # Левый край
                    cumulative[i, j] += min(cumulative[i-1, j], cumulative[i-1, j+1])
                elif j == width - 1:  # Правый край
                    cumulative[i, j] += min(cumulative[i-1, j-1], cumulative[i-1, j])
                else:  # Середина
                    cumulative[i, j] += min(cumulative[i-1, j-1], cumulative[i-1, j], cumulative[i-1, j+1])
        
        return cumulative
    
    def cumulative_map_forward(self, image, energy_map):
        """Вычисляет кумулятивную карту энергии для алгоритма forward energy"""
        # Создаем ядра для вычисления матриц соседей
        kernel_x = np.array([[0., 0., 0.], [-1., 0., 1.], [0., 0., 0.]], dtype=np.float64)
        kernel_y_left = np.array([[0., 0., 0.], [0., 0., 1.], [0., -1., 0.]], dtype=np.float64)
        kernel_y_right = np.array([[0., 0., 0.], [1., 0., 0.], [0., -1., 0.]], dtype=np.float64)
        
        # Вычисляем матрицы соседей
        matrix_x = self.calc_neighbor_matrix(image, kernel_x)
        matrix_y_left = self.calc_neighbor_matrix(image, kernel_y_left)
        matrix_y_right = self.calc_neighbor_matrix(image, kernel_y_right)
        
        # Преобразуем все в float64 для предотвращения переполнения
        energy_map = energy_map.astype(np.float64)
        matrix_x = matrix_x.astype(np.float64)
        matrix_y_left = matrix_y_left.astype(np.float64)
        matrix_y_right = matrix_y_right.astype(np.float64)
        
        m, n = energy_map.shape
        output = np.copy(energy_map)
        
        # Заполняем кумулятивную карту энергии с учетом forward energy
        for row in range(1, m):
            for col in range(n):
                if col == 0:  # Левый край
                    e_right = float(output[row - 1, col + 1]) + float(matrix_x[row - 1, col + 1]) + float(matrix_y_right[row - 1, col + 1])
                    e_up = float(output[row - 1, col]) + float(matrix_x[row - 1, col])
                    output[row, col] = float(energy_map[row, col]) + min(e_right, e_up)
                elif col == n - 1:  # Правый край
                    e_left = float(output[row - 1, col - 1]) + float(matrix_x[row - 1, col - 1]) + float(matrix_y_left[row - 1, col - 1])
                    e_up = float(output[row - 1, col]) + float(matrix_x[row - 1, col])
                    output[row, col] = float(energy_map[row, col]) + min(e_left, e_up)
                else:  # Середина
                    e_left = float(output[row - 1, col - 1]) + float(matrix_x[row - 1, col - 1]) + float(matrix_y_left[row - 1, col - 1])
                    e_right = float(output[row - 1, col + 1]) + float(matrix_x[row - 1, col + 1]) + float(matrix_y_right[row - 1, col + 1])
                    e_up = float(output[row - 1, col]) + float(matrix_x[row - 1, col])
                    output[row, col] = float(energy_map[row, col]) + min(e_left, e_right, e_up)
        
        return output
    
    def calc_neighbor_matrix(self, image, kernel):
        """Вычисляет матрицу соседей с помощью фильтра"""
        b = image[:, :, 0]
        g = image[:, :, 1]
        r = image[:, :, 2]
        
        output = np.absolute(ndimage.convolve(b, kernel)) + \
                 np.absolute(ndimage.convolve(g, kernel)) + \
                 np.absolute(ndimage.convolve(r, kernel))
        return output
    
    def find_min_seam(self, cumulative):
        """Находит шов с минимальной энергией"""
        height, width = cumulative.shape
        # Массив для хранения индексов шва
        seam = np.zeros(height, dtype=np.int32)
        
        # Находим минимальный элемент в последней строке
        seam[-1] = np.argmin(cumulative[-1])
        
        # Идем снизу вверх и находим путь с минимальной энергией
        for i in range(height-2, -1, -1):
            j = seam[i+1]
            # Обрабатываем крайние случаи
            if j == 0:
                seam[i] = np.argmin(cumulative[i, 0:2])
            elif j == width-1:
                seam[i] = width-1 + np.argmin(cumulative[i, width-2:width]) - 1
            else:
                seam[i] = j + np.argmin(cumulative[i, j-1:j+2]) - 1
        
        return seam
    
    def remove_seam(self, image, seam):
        """Удаляет шов из изображения"""
        height, width, channels = image.shape
        # Создаем новое изображение без шва
        result = np.zeros((height, width-1, channels), dtype=image.dtype)
        
        # Для каждой строки удаляем пиксель шва
        for i in range(height):
            # Индекс пикселя шва в текущей строке
            j = seam[i]
            # Копируем пиксели до шва
            result[i, :j, :] = image[i, :j, :]
            # Копируем пиксели после шва
            result[i, j:, :] = image[i, j+1:, :]
        
        return result
    
    def duplicate_seam(self, image, seam):
        """Дублирует шов в изображении с плавной интерполяцией"""
        height, width, channels = image.shape
        # Создаем новое изображение с дополнительным столбцом
        result = np.zeros((height, width+1, channels), dtype=image.dtype)
        
        # Для каждой строки дублируем пиксель шва с плавной интерполяцией
        for i in range(height):
            # Индекс пикселя шва в текущей строке
            j = seam[i]
            
            # Копируем пиксели до шва
            result[i, :j, :] = image[i, :j, :]
            
            # Интерполируем пиксели вокруг шва для более плавного перехода
            if j == 0:  # Левый край
                result[i, j, :] = image[i, j, :]
                result[i, j+1, :] = image[i, j, :]
            elif j == width-1:  # Правый край
                result[i, j, :] = image[i, j, :]
                result[i, j+1, :] = image[i, j, :]
            else:  # Середина
                # Сохраняем текущий пиксель
                result[i, j, :] = image[i, j, :]
                
                # Создаем новый пиксель с плавной интерполяцией
                # Используем взвешенное среднее из соседних пикселей
                left = image[i, max(0, j-1), :].astype(np.int32)
                current = image[i, j, :].astype(np.int32)
                right = image[i, min(width-1, j+1), :].astype(np.int32)
                
                # Вычисляем взвешенное среднее и преобразуем обратно в исходный тип
                new_pixel = (left * 25 + current * 50 + right * 25) // 100
                result[i, j+1, :] = new_pixel.astype(image.dtype)
            
            # Копируем пиксели после шва
            result[i, j+2:, :] = image[i, j+1:, :]
        
        # Сглаживаем область вокруг вставленных швов
        result = self.smooth_seam_area(result, seam)
        
        return result
    
    def update_seams(self, remaining_seams, current_seam):
        """Обновляет индексы оставшихся швов после добавления нового"""
        updated_seams = []
        for seam in remaining_seams:
            # Создаем копию шва
            updated_seam = seam.copy()
            # Увеличиваем индексы для всех позиций, которые идут после добавленного шва
            for i in range(len(updated_seam)):
                if updated_seam[i] >= current_seam[i]:
                    updated_seam[i] += 1
            updated_seams.append(updated_seam)
        return updated_seams
        
    def smooth_seam_area(self, image, seam):
        """Сглаживает область вокруг вставленного шва для устранения артефактов"""
        height, width, channels = image.shape
        result = np.copy(image)
        
        # Для каждой строки сглаживаем область вокруг шва
        for i in range(height):
            j = seam[i] + 1  # Индекс вставленного пикселя
            
            # Пропускаем края изображения
            if j <= 1 or j >= width - 1:
                continue
                
            # Применяем локальное сглаживание в области шва
            # Используем окно 3x3 для сглаживания
            for c in range(channels):
                # Сглаживаем только вставленный пиксель и его соседей
                window = np.array([image[i, j-1, c], image[i, j, c], image[i, j+1, c]])
                # Применяем медианный фильтр для удаления выбросов
                median_value = np.median(window).astype(image.dtype)
                result[i, j, c] = median_value
        
        return result
        
    def apply_smooth_filter(self, image):
        """Применяет легкое размытие для сглаживания артефактов"""
        # Используем медианный фильтр вместо гауссова размытия для сохранения яркости
        # Применяем фильтр только к областям с высокой частотой (швам)
        
        # Создаем маску краев (потенциальных швов)
        edges = np.zeros_like(image[:,:,0], dtype=bool)
        
        # Используем оператор Собеля для обнаружения краев
        for channel in range(image.shape[2]):
            dx = ndimage.sobel(image[:,:,channel], axis=1)
            dy = ndimage.sobel(image[:,:,channel], axis=0)
            edge_strength = np.sqrt(dx**2 + dy**2)
            # Выбираем только сильные края (потенциальные швы)
            threshold = np.percentile(edge_strength, 95)  # Верхние 5% краев
            edges = edges | (edge_strength > threshold)
        
        # Расширяем маску краев для захвата окрестности
        edges = ndimage.binary_dilation(edges, iterations=1)
        
        # Применяем медианный фильтр только к обнаруженным краям
        result = image.copy()
        for channel in range(image.shape[2]):
            channel_data = image[:,:,channel].copy()
            filtered = ndimage.median_filter(channel_data, size=3)
            # Применяем фильтр только к краям
            channel_data[edges] = filtered[edges]
            result[:,:,channel] = channel_data
        
        return result
    
    def rotate_image(self, image, ccw):
        """Поворачивает изображение на 90 градусов"""
        m, n, ch = image.shape
        output = np.zeros((n, m, ch), dtype=image.dtype)
        
        if ccw:  # Против часовой стрелки
            # Поворот на 90 градусов против часовой стрелки
            output = np.transpose(image, (1, 0, 2))[::-1]
        else:  # По часовой стрелке
            # Поворот на 90 градусов по часовой стрелке
            output = np.transpose(image, (1, 0, 2))[:, ::-1]
        
        return output

###########################################################################################################

    def reset_gamma(self):
        """Сбросить изменения гаммы"""
        if self.temp_image:
            self.image = self.temp_image.copy()
            self.temp_image = None
            self.gamma_slider.setValue(100)  # Возвращаем слайдер к 1.0
            self.current_gamma = 1.0
            self.gamma_label.setText("γ = 1.00")

            # Обновляем отображение
            self.display_image()
            self.display_histogram()

    def apply_gamma(self):
        """Применить текущую гамма-коррекцию и добавить в историю"""
        if self.temp_image:
            # Добавляем исходное изображение в историю
            self.history.append(self.temp_image.copy())
            self.redo_stack.clear()
            self.temp_image = None

            # Добавляем запись в историю действий
            self.history_list.addItem(f"Гамма-коррекция: γ={self.current_gamma:.2f}")
            print(f"Гамма-коррекция применена: γ={self.current_gamma:.2f}")

    def apply_gamma_correction(self):
        """Оригинальная функция с диалогом для опытных пользователей"""
        if not self.image:
            return

        gamma, ok = QInputDialog.getDouble(
            self, "Гамма-коррекция",
            "Введите значение гаммы (0.1 – 5.0):",
            value=1.0, min=0.1, max=5.0, decimals=2
        )

        if not ok:
            return

        # сохраняем в историю и сбрасываем временное изображение
        self.history.append(self.image.copy())
        self.redo_stack.clear()
        self.temp_image = None

        img_array = np.array(self.image).astype(np.float32) / 255.0
        corrected = np.power(img_array, 1.0 / gamma)
        corrected = (corrected * 255).clip(0, 255).astype(np.uint8)
        self.image = Image.fromarray(corrected)

        # обновляем слайдер, чтобы он соответствовал новому значению
        self.gamma_slider.setValue(int(gamma * 100))
        self.current_gamma = gamma
        self.gamma_label.setText(f"γ = {gamma:.2f}")

        self.display_image()
        self.display_histogram()
        self.history_list.addItem(f"Гамма-коррекция: γ={gamma:.2f}")

###########################################################################################################

    def display_image(self):
        if self.image:
            q_image = ImageQt(self.image)
            pixmap = QPixmap.fromImage(q_image)

            # применяем масштаб к реальным размерам изображения
            img_width, img_height = self.image.size
            scaled_width = int(img_width * self.scale_factor)
            scaled_height = int(img_height * self.scale_factor)

            if scaled_width > 0 and scaled_height > 0:
                # масштабируем изображение с указанным коэффициентом
                scaled_pixmap = pixmap.scaled(
                    scaled_width,
                    scaled_height,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )

                # новый pixmap с дополнительным пространством для пунктирной рамки
                final_pixmap = QPixmap(scaled_width + 2, scaled_height + 2)
                final_pixmap.fill(Qt.GlobalColor.transparent)

                # рисуем масштабированное изображение на новом pixmap
                painter = QPainter(final_pixmap)
                painter.drawPixmap(1, 1, scaled_pixmap)

                # пунктирная рамка
                pen = QPen(Qt.GlobalColor.white)
                pen.setStyle(Qt.PenStyle.DashLine)
                painter.setPen(pen)
                painter.drawRect(0, 0, scaled_width + 1, scaled_height + 1)
                painter.end()

                # установка pixmap в метку
                self.drawing_area.setPixmap(final_pixmap)
                self.drawing_area.resize(final_pixmap.size())

                # обновление статусбара с процентом масштаба
                scale_percent = int(self.scale_factor * 100)
                self.update_statusbar(scale_percent=scale_percent)

                print(f"Изображение отображено. Масштаб: {scale_percent}%")

    def update_statusbar(self, scale_percent=None):
        if self.image:
            width, height = self.image.size
            if scale_percent is None:
                scale_percent = int(self.scale_factor * 100)
            self.statusBar().showMessage(f"Размер изображения: {width}x{height} | Масштаб: {scale_percent}%")
        else:
            self.statusBar().showMessage("Нет изображения")

###########################################################################################################

    def zoom_in(self):
        self.set_scale(self.scale_factor * 1.25)  # Увеличиваем на 25%

    def zoom_out(self):
        self.set_scale(self.scale_factor * 0.8)  # Уменьшаем на 20%

    def zoom_original(self):
        self.set_scale(1.0)  # 100% масштаб

    def set_scale(self, scale):
        if scale > 0:
            self.scale_factor = scale
            self.display_image()
            self.history_list.addItem(f"Масштаб изменен: {int(scale * 100)}%")

    def modify_image(self, func, *args):
        if self.image:
            try:
                self.history.append(self.image.copy())
                self.redo_stack.clear()
                func(*args)
                self.display_image()
                self.display_histogram()
                self.history_list.addItem(f"{self.get_action_description(func)}")
                print(f"Изменение выполнено, стек истории: {len(self.history)} записей.")
            except Exception as e:
                print(f"Ошибка при изменении изображения: {e}")
                # Восстанавливаем последнее изображение из истории
                if self.history:
                    self.image = self.history[-1]
                    self.display_image()
                    self.display_histogram()

###########################################################################################################

    def get_action_description(self, func):
        descriptions = {
            self.rotate_image: "Поворот изображения",
            self.flip_horizontal: "Отражение по горизонтали",
            self.flip_vertical: "Отражение по вертикали",
            self.apply_grayscale: "Применение фильтра серого",
            self.apply_invert: "Инверсия цветов",
            self.apply_binary: "Бинаризация",
            self.adjust_image_scale: "Масштабирование с учетом содержимого",
            self.apply_gamma_correction: "Гамма-коррекция",
            self.content_aware_resize: "Масштабирование с учетом содержимого",
            self.crop_image: "Кадрирование изображения"
        }
        return descriptions.get(func, "Неизвестное действие")

    def add_history_item(self, text, icon_path=None, color=None):
        """Добавляет элемент в историю действий с форматированием"""
        item = QListWidgetItem()

        # добавляем текущее время к записи
        from datetime import datetime
        current_time = datetime.now().strftime("%H:%M:%S")
        formatted_text = f"{text} [{current_time}]"

        item.setText(formatted_text)

        if icon_path:
            item.setIcon(QIcon(icon_path))
        elif "Поворот" in text:
            item.setIcon(QIcon("icons/rotate.png"))
        elif "Отражение" in text:
            item.setIcon(QIcon("icons/flip.png"))
        elif "Масштаб" in text or "размер" in text:
            item.setIcon(QIcon("icons/resize.png"))
        elif "Гамма" in text:
            item.setIcon(QIcon("icons/gamma.png"))
        elif "Открыт файл" in text:
            item.setIcon(QIcon("icons/open.png"))
        elif "Сохранён" in text:
            item.setIcon(QIcon("icons/save.png"))
        elif "Отменено" in text:
            item.setIcon(QIcon("icons/undo.png"))
        elif "Восстановлено" in text:
            item.setIcon(QIcon("icons/redo.png"))

        if color:
            item.setBackground(QBrush(QColor(color)))

        # устанавливаем выравнивание и размер шрифта
        item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        # вставляем в начало списка
        self.history_list.insertItem(0, item)

        # прокручиваем к началу списка
        self.history_list.scrollToTop()

        # если история скрыта, устанавливаем флаг новых элементов и обновляем текст меню
        if not self.history_group.isVisible():
            self.has_new_history_items = True
            self.update_history_menu_text()
            print("Добавлено новое действие в историю")

    def update_history_display(self):
        """Обновляет отображение истории, прокручивая к первому элементу"""
        if self.history_list.count() > 0:
            self.history_list.scrollToItem(self.history_list.item(0))

    def update_history_menu_text(self):
        """Обновляет текст пункта меню истории действий, добавляя индикатор новых элементов"""
        actions = self.get_actions()
        if self.has_new_history_items:
            actions['history'].setText(f"{self.history_action_text} (•)")
        else:
            actions['history'].setText(self.history_action_text)

    def toggle_history_panel(self):
        """Переключает видимость панели истории действий"""
        if self.history_group.isVisible():
            self.hide_history_panel()
        else:
            self.show_history_panel()

    def show_history_panel(self):
        """Показывает панель истории действий"""
        self.history_group.setVisible(True)
        # Сбрасываем флаг новых элементов и обновляем текст меню
        self.has_new_history_items = False
        self.update_history_menu_text()
        print("Панель истории действий отображена")

    def hide_history_panel(self):
        """Скрывает панель истории действий"""
        self.history_group.setVisible(False)
        print("Панель истории действий скрыта")

    def undo_action(self):
        if len(self.history) > 1:
            self.redo_stack.append(self.image.copy())
            self.image = self.history.pop()
            self.display_image()
            self.display_histogram()
            self.history_list.addItem("Действие отменено.")

    def redo_action(self):
        if self.redo_stack:
            self.history.append(self.image.copy())
            self.image = self.redo_stack.pop()
            self.display_image()
            self.display_histogram()
            self.history_list.addItem("Действие восстановлено.")

    def rotate_image(self, angle):
        if self.image:
            self.history.append(self.image.copy())
            self.redo_stack.clear()
            self.image = self.image.rotate(angle, expand=True)
            self.display_image()
            self.display_histogram()
            self.history_list.addItem(f"Поворот на {angle}°")

    def adjust_image_scale(self):
        if self.image:
            screen_width = self.scroll_area.width()
            screen_height = self.scroll_area.height()
            img_width, img_height = self.image.size

            if img_width > screen_width or img_height > screen_height:
                scale_factor = min(screen_width / img_width, screen_height / img_height)
                new_width = int(img_width * scale_factor)
                new_height = int(img_height * scale_factor)
                self.image = self.image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                print("Изображение масштабировано для соответствия области отображения.")


    def rotate_left(self):
        self.rotate_image(-90)

    def rotate_right(self):
        self.rotate_image(90)

    def flip_horizontal(self):
        if self.image:
            self.image = ImageOps.mirror(self.image)

    def flip_vertical(self):
        if self.image:
            self.image = ImageOps.flip(self.image)

    def apply_grayscale(self):
        if self.image:
            self.image = self.image.convert("L").convert("RGB")

    def apply_invert(self):
        if self.image:
            self.image = ImageOps.invert(self.image.convert("RGB"))

    def apply_binary(self):
        if self.image:
            grayscale_image = self.image.convert("L")
            self.image = grayscale_image.point(lambda x: 0 if x < 128 else 255, mode='1').convert("RGB")

###########################################################################################################

    def toggle_theme(self):
        """Переключает между светлой и темной темой"""
        if self.current_theme == "dark":
            self.current_theme = "light"
        else:
            self.current_theme = "dark"

        # применяем выбранную тему
        self.apply_theme()

        # обновляем гистограмму, если изображение загружено
        if self.image:
            self.display_histogram()

        # добавляем запись в историю действий
        theme_name = "светлую" if self.current_theme == "light" else "темную"
        self.history_list.addItem(f"Переключение на {theme_name} тему")

        # обновляем статусбар
        self.update_statusbar()

    def apply_theme(self):
        """Применяет текущую тему и обновляет гистограмму"""
        # вызываем метод базового класса для применения основных стилей
        super().apply_theme()

        # дополнительные настройки для гистограммы
        if self.image:
            self.display_histogram()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EditorLogic()
    window.show()
    sys.exit(app.exec())