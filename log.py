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
        """Реализация алгоритма Seam Carving для масштабирования с учетом содержимого"""
        img_array = np.array(img)
        current_height, current_width = img_array.shape[:2]

        # Определяем, насколько нужно изменить размеры
        width_diff = target_width - current_width
        height_diff = target_height - current_height

        # Изменяем ширину
        if width_diff < 0:  # Уменьшаем ширину
            for i in range(abs(width_diff)):
                # Находим и удаляем вертикальный шов
                energy = self.compute_energy(img_array)
                seam = self.find_vertical_seam(energy)
                img_array = self.remove_vertical_seam(img_array, seam)
        elif width_diff > 0:  # Увеличиваем ширину (простая реализация)
            # Находим и сохраняем швы для последующего дублирования
            seams = []
            temp_img = np.copy(img_array)
            for i in range(min(width_diff, 50)):  # Ограничение для производительности
                energy = self.compute_energy(temp_img)
                seam = self.find_vertical_seam(energy)
                seams.append(seam)
                temp_img = self.remove_vertical_seam(temp_img, seam)

            # Дублируем швы в обратном порядке (от наименее важных к более важным)
            for seam in reversed(seams):
                img_array = self.duplicate_vertical_seam(img_array, seam)

        # Изменяем высоту (аналогично ширине, но с транспонированным изображением)
        if height_diff < 0:  # Уменьшаем высоту
            for i in range(abs(height_diff)):
                img_array = np.transpose(img_array, (1, 0, 2)) if len(img_array.shape) == 3 else np.transpose(img_array)
                energy = self.compute_energy(img_array)
                seam = self.find_vertical_seam(energy)
                img_array = self.remove_vertical_seam(img_array, seam)
                img_array = np.transpose(img_array, (1, 0, 2)) if len(img_array.shape) == 3 else np.transpose(img_array)
        elif height_diff > 0:  # Увеличиваем высоту
            seams = []
            temp_img = np.copy(img_array)
            temp_img = np.transpose(temp_img, (1, 0, 2)) if len(temp_img.shape) == 3 else np.transpose(temp_img)

            for i in range(min(height_diff, 50)):
                energy = self.compute_energy(temp_img)
                seam = self.find_vertical_seam(energy)
                seams.append(seam)
                temp_img = self.remove_vertical_seam(temp_img, seam)

            img_array = np.transpose(img_array, (1, 0, 2)) if len(img_array.shape) == 3 else np.transpose(img_array)
            for seam in reversed(seams):
                img_array = self.duplicate_vertical_seam(img_array, seam)
            img_array = np.transpose(img_array, (1, 0, 2)) if len(img_array.shape) == 3 else np.transpose(img_array)

        return Image.fromarray(img_array)

    def compute_energy(self, img):
        """Вычисляет энергию изображения с помощью оператора Собеля"""
        if len(img.shape) == 3:
            # Преобразуем RGB в grayscale для вычисления энергии
            gray_img = np.mean(img, axis=2).astype(np.float64)
        else:
            gray_img = img.astype(np.float64)

        # Вычисляем градиенты с помощью оператора Собеля
        sobelx = ndimage.sobel(gray_img, axis=1)
        sobely = ndimage.sobel(gray_img, axis=0)

        # Общая энергия - сумма абсолютных значений градиентов
        return np.abs(sobelx) + np.abs(sobely)

    def find_vertical_seam(self, energy):
        """Находит вертикальный шов с минимальной энергией"""
        height, width = energy.shape

        # Матрица кумулятивной энергии для динамического программирования
        cumulative_energy = np.copy(energy)
        for i in range(1, height):
            for j in range(width):
                # Выбираем минимум из трех возможных предыдущих пикселей
                if j == 0:
                    cumulative_energy[i, j] += min(cumulative_energy[i - 1, j], cumulative_energy[i - 1, j + 1])
                elif j == width - 1:
                    cumulative_energy[i, j] += min(cumulative_energy[i - 1, j - 1], cumulative_energy[i - 1, j])
                else:
                    cumulative_energy[i, j] += min(cumulative_energy[i - 1, j - 1],
                                                   cumulative_energy[i - 1, j],
                                                   cumulative_energy[i - 1, j + 1])

        # Восстановление пути (шва)
        seam = np.zeros(height, dtype=np.int32)

        # Находим индекс минимума в последней строке
        seam[-1] = np.argmin(cumulative_energy[-1])

        # Восстанавливаем путь снизу вверх
        for i in range(height - 2, -1, -1):
            j = seam[i + 1]

            if j == 0:
                seam[i] = j if cumulative_energy[i, j] <= cumulative_energy[i, j + 1] else j + 1
            elif j == width - 1:
                seam[i] = j - 1 if cumulative_energy[i, j - 1] <= cumulative_energy[i, j] else j
            else:
                neighbors = [cumulative_energy[i, j - 1], cumulative_energy[i, j], cumulative_energy[i, j + 1]]
                min_idx = np.argmin(neighbors)
                seam[i] = j + (min_idx - 1)  # -1, 0 или +1

        return seam

    def remove_vertical_seam(self, img, seam):
        """Удаляет вертикальный шов из изображения"""
        height, width = img.shape[:2]

        # Создаем новое изображение без шва
        if len(img.shape) == 3:
            new_img = np.zeros((height, width - 1, img.shape[2]), dtype=img.dtype)
        else:
            new_img = np.zeros((height, width - 1), dtype=img.dtype)

        # Копируем все пиксели, кроме шва
        for i in range(height):
            j_seam = seam[i]
            if len(img.shape) == 3:
                new_img[i, :j_seam] = img[i, :j_seam]
                new_img[i, j_seam:] = img[i, j_seam + 1:]
            else:
                new_img[i, :j_seam] = img[i, :j_seam]
                new_img[i, j_seam:] = img[i, j_seam + 1:]

        return new_img

    def duplicate_vertical_seam(self, img, seam):
        """Дублирует вертикальный шов для увеличения размера"""
        height, width = img.shape[:2]

        # Создаем новое изображение с дополнительным столбцом
        if len(img.shape) == 3:
            new_img = np.zeros((height, width + 1, img.shape[2]), dtype=img.dtype)
        else:
            new_img = np.zeros((height, width + 1), dtype=img.dtype)

        # Вставляем дублированные пиксели
        for i in range(height):
            j_seam = seam[i]

            if len(img.shape) == 3:
                new_img[i, :j_seam] = img[i, :j_seam]
                new_img[i, j_seam] = img[i, j_seam]  # Дублируем пиксель
                new_img[i, j_seam + 1:] = img[i, j_seam:]
            else:
                new_img[i, :j_seam] = img[i, :j_seam]
                new_img[i, j_seam] = img[i, j_seam]
                new_img[i, j_seam + 1:] = img[i, j_seam:]

        return new_img

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