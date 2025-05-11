from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QLabel, QMenuBar, QMenu,
    QStatusBar, QWidget, QListWidget, QPushButton, QFrame, QSizePolicy, QSplitter,
    QSlider, QGroupBox, QScrollArea
)
from PyQt6.QtGui import QAction, QFont, QFontDatabase, QIcon
from PyQt6.QtCore import Qt, QSize


class GraphicEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Графический редактор")
        self.setGeometry(100, 100, 1200, 800)
        self.current_theme = "dark"  # добавляем переменную для отслеживания текущей темы
        self.setStyleSheet("background-color: #2E2E2E; color: #F0F0F0;")

        font_id = QFontDatabase.addApplicationFont("fonts/Roboto-Regular.ttf")
        if font_id != -1:
            font_family = QFontDatabase.applicationFontFamilies(font_id)[0]
            Roboto_font = QFont(font_family, 10)
            self.setFont(Roboto_font)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        top_layout = QHBoxLayout()
        self.scroll_area = QScrollArea()
        self.scroll_area.setStyleSheet("background-color: #4A4A4A;")
        self.scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll_area.setWidgetResizable(False)  # важно для работы масштабирования

        self.drawing_area = QLabel()
        self.drawing_area.setStyleSheet("background-color: #4A4A4A;")
        self.drawing_area.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.scroll_area.setWidget(self.drawing_area)
        top_layout.addWidget(self.scroll_area, 3)

        right_panel = QVBoxLayout()

        # История действий
        self.right_splitter = QSplitter(Qt.Orientation.Vertical)
        self.right_splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #555555;
                height: 2px;
            }
            QSplitter::handle:hover {
                background-color: #999999;
            }
        """)

        # кнопка закрытия для истории действий
        history_header_layout = QHBoxLayout()
        history_header_layout.setContentsMargins(0, 0, 0, 4)  # Уменьшаем отступ снизу
        self.history_title = QLabel("История действий")  # Сохраняем как атрибут класса
        self.history_title.setStyleSheet("color: #F0F0F0; font-size: 12px; font-weight: bold;")
        self.history_title.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)  # Выравнивание по левому краю
        history_header_layout.addWidget(self.history_title)

        # spacer для отступа
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        history_header_layout.addWidget(spacer)

        # кнопка закрытия
        self.close_history_button = QPushButton("✕")
        self.close_history_button.setFixedSize(16, 16)
        self.close_history_button.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #AAAAAA;
                border: none;
                font-size: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                color: #FFFFFF;
            }
            QPushButton:pressed {
                color: #888888;
            }
        """)
        history_header_layout.addWidget(self.close_history_button)

        # вертикальный лейаут для GroupBox
        history_layout = QVBoxLayout()
        history_layout.setContentsMargins(8, 8, 8, 8)
        history_layout.setSpacing(4)

        # заголовок с кнопкой закрытия
        history_layout.addLayout(history_header_layout)

        # QListWidget для истории
        self.history_list = QListWidget()
        self.history_list.setStyleSheet("""
            QListWidget {
                background-color: #3A3A3A; 
                color: #F0F0F0; 
                font-size: 12px; 
                border: none;
                padding: 2px;
            }
            QListWidget::item {
                padding: 4px 2px;
                border-bottom: 1px solid #444444;
            }
            QListWidget::item:selected {
                background-color: #555555;
            }
            QScrollBar:vertical {
                border: none;
                background: #333333;
                width: 8px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #666666;
                min-height: 20px;
                border-radius: 4px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        self.history_list.setMinimumHeight(120)  # Увеличиваем минимальную высоту
        self.history_list.setWordWrap(True)  # Включаем перенос текста
        self.history_list.setAlternatingRowColors(True)  # Чередующиеся цвета строк
        self.history_list.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)  # Отключаем горизонтальную прокрутку
        self.history_list.setVerticalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)  # Плавная прокрутка

        history_layout.addWidget(self.history_list)

        # GroupBox для истории
        self.history_group = QGroupBox()
        self.history_group.setStyleSheet("""
            QGroupBox {
                background-color: #3A3A3A;
                color: #F0F0F0;
                border: 2px solid #4A4A4A;
                border-radius: 8px;
                margin-top: 8px;
                font-size: 12px;
                font-weight: normal;
            }
        """)
        self.history_group.setLayout(history_layout)
        self.history_group.setMinimumWidth(250)  # Устанавливаем минимальную ширину
        self.history_group.setMaximumWidth(350)  # Устанавливаем максимальную ширину

        # скрываем панель истории при запуске
        self.history_group.setVisible(False)

        # добавляем группу истории в правую панель
        right_panel.addWidget(self.history_group)

        # контейнер для остальных элементов
        self.other_widgets = QWidget()
        other_layout = QVBoxLayout(self.other_widgets)
        other_layout.setContentsMargins(0, 0, 0, 0)

        self.right_splitter.addWidget(self.other_widgets)
        right_panel.addWidget(self.right_splitter)

        # Гамма-коррекция
        self.gamma_group = QGroupBox("Гамма-коррекция")
        self.gamma_group.setStyleSheet("background-color: #3A3A3A; color: #F0F0F0;")
        gamma_layout = QVBoxLayout()
        gamma_control_layout = QHBoxLayout()
        self.gamma_label = QLabel("γ = 1.00")
        self.gamma_label.setStyleSheet("color: #F0F0F0; font-size: 12px;")
        self.gamma_slider = QSlider(Qt.Orientation.Horizontal)
        self.gamma_slider.setRange(10, 500)
        self.gamma_slider.setValue(100)
        self.gamma_slider.setStyleSheet("""
        QSlider::groove:horizontal {
            background: #555555;
            height: 8px;
            border-radius: 4px;
        }
        QSlider::handle:horizontal {
            background: #D0D0D0;
            width: 16px;
            margin: -4px 0;
            border-radius: 8px;
        }
        """)
        gamma_control_layout.addWidget(self.gamma_slider, 4)
        gamma_control_layout.addWidget(self.gamma_label, 1)
        gamma_layout.addLayout(gamma_control_layout)
        gamma_buttons_layout = QHBoxLayout()
        self.reset_gamma_button = QPushButton("Сброс")
        self.reset_gamma_button.setStyleSheet("background-color: #4A4A4A; color: #F0F0F0;")
        self.apply_gamma_button = QPushButton("Применить")
        self.apply_gamma_button.setStyleSheet("background-color: #4A4A4A; color: #F0F0F0;")
        gamma_buttons_layout.addWidget(self.reset_gamma_button)
        gamma_buttons_layout.addWidget(self.apply_gamma_button)
        gamma_layout.addLayout(gamma_buttons_layout)
        self.gamma_group.setLayout(gamma_layout)
        # right_panel.addWidget(self.gamma_group)
        other_layout.addWidget(self.gamma_group)

        # Гистограмма
        self.histogram_group = QGroupBox("Гистограмма яркости")
        self.histogram_group.setStyleSheet("""
            QGroupBox {
                background-color: #3A3A3A;
                color: #F0F0F0;
                border: 2px solid #4A4A4A;
                border-radius: 8px;
                margin-top: 8px;
                font-size: 12px;
                font-weight: normal;
                padding-top: 12px;
            }
            QGroupBox:title {
                subcontrol-origin: margin;
                left: 10px;
                top: 2px;
                padding: 0 3px;
            }
        """)
        hist_layout = QVBoxLayout()
        hist_layout.setContentsMargins(8, 16, 8, 8)
        self.histogram_label = QLabel()
        self.histogram_label.setMinimumHeight(180)  # Увеличиваем минимальную высоту
        self.histogram_label.setStyleSheet("background-color: #232323; border: none;")
        self.histogram_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hist_layout.addWidget(self.histogram_label)
        self.histogram_group.setLayout(hist_layout)
        # right_panel.addWidget(self.histogram_group)
        other_layout.addWidget(self.histogram_group)

        # Кнопки отмены/повтора
        buttons_layout = QHBoxLayout()
        self.undo_button = QPushButton()
        self.undo_button.setIcon(QIcon("icons/undo.png"))
        self.undo_button.setIconSize(QSize(18, 18))
        self.undo_button.setToolTip("Отменить")
        self.undo_button.setFixedSize(32, 32)
        self.undo_button.setStyleSheet("""
                QPushButton {
                    background-color: #3A3A3A;
                    color: #F0F0F0;
                    border: 2px solid #4A4A4A;
                    border-radius: 8px;
                    padding: 2px;
                }
                QPushButton:hover {
                    background-color: #444444;
                    border-color: #888888;
                }
                QPushButton:pressed {
                    background-color: #222222;
                }
            """)
        buttons_layout.addWidget(self.undo_button)

        self.redo_button = QPushButton()
        self.redo_button.setIcon(QIcon("icons/redo.png"))
        self.redo_button.setIconSize(QSize(18, 18))
        self.redo_button.setToolTip("Повторить")
        self.redo_button.setFixedSize(32, 32)
        self.redo_button.setStyleSheet("""
                QPushButton {
                    background-color: #3A3A3A;
                    color: #F0F0F0;
                    border: 2px solid #4A4A4A;
                    border-radius: 8px;
                    padding: 2px;
                }
                QPushButton:hover {
                    background-color: #444444;
                    border-color: #888888;
                }
                QPushButton:pressed {
                    background-color: #222222;
                }
            """)
        buttons_layout.addWidget(self.redo_button)

        other_layout.addLayout(buttons_layout)
        # right_panel.addLayout(buttons_layout, 1)

        top_layout.addLayout(right_panel, 1)
        main_layout.addLayout(top_layout, 9)

######################################################################################

        menu = self.menuBar()

        file_menu = menu.addMenu("Файл")
        open_action = QAction("Открыть", self)
        open_action.setObjectName("open_action")
        save_action = QAction("Сохранить", self)
        save_action.setObjectName("save_action")
        save_action.setShortcut("Ctrl+S")  # Горячая клавиша Ctrl+S для сохранения
        file_menu.addAction(open_action)
        file_menu.addAction(save_action)

        edit_menu = menu.addMenu("Правка")
        undo_action = QAction("Отменить", self)
        undo_action.setObjectName("undo_action")
        undo_action.setShortcut("Ctrl+Z")  # Горячая клавиша Ctrl+Z для отмены
        redo_action = QAction("Повторить", self)
        redo_action.setObjectName("redo_action")
        redo_action.setShortcut("Ctrl+Y")  # Горячая клавиша Ctrl+Y для повтора

        # Добавляем действие для отображения истории
        self.history_action_text = "История действий"
        history_action = QAction(self.history_action_text, self)
        history_action.setObjectName("history_action")
        history_action.setShortcut("Ctrl+H")  # Добавляем горячую клавишу Ctrl+H

        edit_menu.addAction(undo_action)
        edit_menu.addAction(redo_action)
        edit_menu.addSeparator()  # Разделитель
        edit_menu.addAction(history_action)

        # Добавление меню Просмотр с операциями масштабирования
        view_menu = menu.addMenu("Просмотр")
        zoom_in_action = QAction("Увеличить", self)
        zoom_in_action.setObjectName("zoom_in_action")
        zoom_in_action.setShortcut("Ctrl++")
        zoom_out_action = QAction("Уменьшить", self)
        zoom_out_action.setObjectName("zoom_out_action")
        zoom_out_action.setShortcut("Ctrl+-")
        zoom_original_action = QAction("Исходный размер (100%)", self)
        zoom_original_action.setObjectName("zoom_original_action")
        zoom_original_action.setShortcut("Ctrl+0")
        view_menu.addAction(zoom_in_action)
        view_menu.addAction(zoom_out_action)
        view_menu.addAction(zoom_original_action)

        image_menu = menu.addMenu("Изображение")

        # подменю для поворота
        rotate_menu = QMenu("Поворот", self)

        rotate_right_action = QAction("Поворот на 90° вправо", self)
        rotate_right_action.setObjectName("rotate_right_action")
        rotate_left_action = QAction("Поворот на 90° влево", self)
        rotate_left_action.setObjectName("rotate_left_action")

        # добавляем действия в подменю поворота
        rotate_menu.addAction(rotate_right_action)
        rotate_menu.addAction(rotate_left_action)

        # добавляем подменю поворота в основное меню изображения
        image_menu.addMenu(rotate_menu)

        flip_horizontal_action = QAction("Отразить по горизонтали", self)
        flip_horizontal_action.setObjectName("flip_horizontal_action")
        flip_vertical_action = QAction("Отразить по вертикали", self)
        flip_vertical_action.setObjectName("flip_vertical_action")
        crop_menu = QMenu("Кадрирование", self)
        crop_square_action = QAction("Квадрат (1:1)", self)
        crop_square_action.setObjectName("crop_square_action")
        crop_4_3_action = QAction("Прямоугольник (4:3)", self)
        crop_4_3_action.setObjectName("crop_4_3_action")
        crop_16_9_action = QAction("Прямоугольник (16:9)", self)
        crop_16_9_action.setObjectName("crop_16_9_action")

        crop_menu.addSeparator()
        crop_menu.addAction(crop_square_action)
        crop_menu.addAction(crop_4_3_action)
        crop_menu.addAction(crop_16_9_action)

        content_aware_action = QAction("Масштабирование с учетом содержимого", self)
        content_aware_action.setObjectName("content_aware_action")
        image_menu.addAction(flip_horizontal_action)
        image_menu.addAction(flip_vertical_action)
        image_menu.addAction(content_aware_action)
        image_menu.addMenu(crop_menu)

        filters_menu = menu.addMenu("Фильтры")
        grayscale_action = QAction("Градации серого", self)
        grayscale_action.setObjectName("grayscale_action")
        invert_action = QAction("Негатив", self)
        invert_action.setObjectName("invert_action")
        binary_action = QAction("Бинаризация", self)
        binary_action.setObjectName("binary_action")
        gamma_correction_action = QAction("Гамма-коррекция", self)
        gamma_correction_action.setObjectName("gamma_correction_action")
        filters_menu.addAction(grayscale_action)
        filters_menu.addAction(invert_action)
        filters_menu.addAction(binary_action)
        filters_menu.addAction(gamma_correction_action)

        help_menu = menu.addMenu("Помощь")
        reference_action = QAction("О программе", self)
        reference_action.setObjectName("reference_action")
        help_menu.addAction(reference_action)

        # Добавляем пункт меню для переключения темы
        self.toggle_theme_action = QAction("Светлая тема", self)
        self.toggle_theme_action.setObjectName("toggle_theme_action")
        help_menu.addAction(self.toggle_theme_action)

        status = QStatusBar()
        status.setStyleSheet("background-color: #3A3A3A; color: #D0D0D0; font-size: 12px;")
        status.showMessage("Изображение отсутствует")
        self.setStatusBar(status)

        self.apply_menu_hover_styles(menu)

    def apply_menu_hover_styles(self, menu):
        menu.setStyleSheet("""
        QMenuBar {
            background-color: #3A3A3A;
            color: #F0F0F0;
            font-family: 'Roboto';
            font-size: 9pt;
        }
        QMenuBar::item {
            background-color: #3A3A3A;
            color: #F0F0F0;
            padding: 5px 15px;
        }
        QMenuBar::item:selected {
            background-color: #555555;
            color: #D0D0D0;
        }
        QMenu::item {
            background-color: #3A3A3A;
            color: #F0F0F0;
            padding: 5px 10px;
        }
        QMenu::item:hover {
            background-color: #555555;
            color: #D0D0D0;
        }
        QMenu::item:selected {
            background-color: #555555;
            color: #D0D0D0;
        }
        """)

    def get_actions(self):
        return {
            'open': self.findChild(QAction, 'open_action'),
            'save': self.findChild(QAction, 'save_action'),
            'rotate_right': self.findChild(QAction, 'rotate_right_action'),
            'rotate_left': self.findChild(QAction, 'rotate_left_action'),
            'flip_horizontal': self.findChild(QAction, 'flip_horizontal_action'),
            'flip_vertical': self.findChild(QAction, 'flip_vertical_action'),
            'grayscale': self.findChild(QAction, 'grayscale_action'),
            'invert': self.findChild(QAction, 'invert_action'),
            'binary': self.findChild(QAction, 'binary_action'),
            'gamma_correction': self.findChild(QAction, 'gamma_correction_action'),
            'content_aware': self.findChild(QAction, 'content_aware_action'),
            'undo': self.findChild(QAction, 'undo_action'),
            'redo': self.findChild(QAction, 'redo_action'),
            'history': self.findChild(QAction, 'history_action'),
            'reference': self.findChild(QAction, 'reference_action'),
            'zoom_in': self.findChild(QAction, 'zoom_in_action'),
            'zoom_out': self.findChild(QAction, 'zoom_out_action'),
            'zoom_original': self.findChild(QAction, 'zoom_original_action'),
            # 'resize_custom': self.findChild(QAction, 'resize_custom_action'),
            'crop_square': self.findChild(QAction, 'crop_square_action'),
            'crop_4_3': self.findChild(QAction, 'crop_4_3_action'),
            'crop_16_9': self.findChild(QAction, 'crop_16_9_action'),
            'content_aware': self.findChild(QAction, 'content_aware_action'),
            'toggle_theme': self.toggle_theme_action,  # Добавляем новое действие для переключения темы
        }

    def apply_theme(self):
        """Применяет стили в зависимости от выбранной темы"""
        if self.current_theme == "dark":
            # Темная тема
            self.setStyleSheet("background-color: #2E2E2E; color: #F0F0F0;")

            # Основные элементы интерфейса
            self.scroll_area.setStyleSheet("background-color: #4A4A4A;")
            self.drawing_area.setStyleSheet("background-color: #4A4A4A;")

            # Панель истории
            if hasattr(self, 'history_group'):
                self.history_group.setStyleSheet("""
                    QGroupBox {
                        background-color: #3A3A3A;
                        color: #F0F0F0;
                        border: 2px solid #4A4A4A;
                        border-radius: 8px;
                        margin-top: 8px;
                        font-size: 12px;
                        font-weight: normal;
                    }
                """)

            if hasattr(self, 'history_list'):
                self.history_list.setStyleSheet("""
                    QListWidget {
                        background-color: #3A3A3A; 
                        color: #F0F0F0; 
                        font-size: 12px; 
                        border: none;
                        padding: 2px;
                    }
                    QListWidget::item {
                        padding: 4px 2px;
                        border-bottom: 1px solid #444444;
                    }
                    QListWidget::item:selected {
                        background-color: #555555;
                    }
                    QScrollBar:vertical {
                        border: none;
                        background: #333333;
                        width: 8px;
                        margin: 0px;
                    }
                    QScrollBar::handle:vertical {
                        background: #666666;
                        min-height: 20px;
                        border-radius: 4px;
                    }
                    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                        height: 0px;
                    }
                """)

            # Заголовок истории действий
            if hasattr(self, 'history_title'):
                self.history_title.setStyleSheet("color: #F0F0F0; font-size: 12px; font-weight: bold;")
            
            # Кнопка закрытия истории
            if hasattr(self, 'close_history_button'):
                self.close_history_button.setStyleSheet("""
                    QPushButton {
                        background-color: transparent;
                        color: #AAAAAA;
                        border: none;
                        font-size: 10px;
                        font-weight: bold;
                    }
                    QPushButton:hover {
                        color: #FFFFFF;
                    }
                    QPushButton:pressed {
                        color: #888888;
                    }
                """)

            # Гамма-коррекция
            if hasattr(self, 'gamma_group'):
                self.gamma_group.setStyleSheet("background-color: #3A3A3A; color: #F0F0F0;")

            if hasattr(self, 'gamma_label'):
                self.gamma_label.setStyleSheet("color: #F0F0F0; font-size: 12px;")

            if hasattr(self, 'gamma_slider'):
                self.gamma_slider.setStyleSheet("""
                    QSlider::groove:horizontal {
                        background: #555555;
                        height: 8px;
                        border-radius: 4px;
                    }
                    QSlider::handle:horizontal {
                        background: #D0D0D0;
                        width: 16px;
                        margin: -4px 0;
                        border-radius: 8px;
                    }
                """)

            if hasattr(self, 'reset_gamma_button') and hasattr(self, 'apply_gamma_button'):
                self.reset_gamma_button.setStyleSheet("background-color: #4A4A4A; color: #F0F0F0;")
                self.apply_gamma_button.setStyleSheet("background-color: #4A4A4A; color: #F0F0F0;")

            # Гистограмма
            if hasattr(self, 'histogram_group'):
                self.histogram_group.setStyleSheet("""
                    QGroupBox {
                        background-color: #3A3A3A;
                        color: #F0F0F0;
                        border: 2px solid #4A4A4A;
                        border-radius: 8px;
                        margin-top: 8px;
                        font-size: 12px;
                        font-weight: normal;
                        padding-top: 12px;
                    }
                    QGroupBox:title {
                        subcontrol-origin: margin;
                        left: 10px;
                        top: 2px;
                        padding: 0 3px;
                    }
                """)

            if hasattr(self, 'histogram_label'):
                self.histogram_label.setStyleSheet("background-color: #232323; border: none;")

            # Кнопки отмены/повтора
            if hasattr(self, 'undo_button') and hasattr(self, 'redo_button'):
                button_style = """
                    QPushButton {
                        background-color: #3A3A3A;
                        color: #F0F0F0;
                        border: 2px solid #4A4A4A;
                        border-radius: 8px;
                        padding: 2px;
                    }
                    QPushButton:hover {
                        background-color: #444444;
                        border-color: #888888;
                    }
                    QPushButton:pressed {
                        background-color: #222222;
                    }
                """
                self.undo_button.setStyleSheet(button_style)
                self.redo_button.setStyleSheet(button_style)

            # Меню
            self.apply_menu_hover_styles(self.menuBar())

            # Статусбар
            if self.statusBar():
                self.statusBar().setStyleSheet("background-color: #3A3A3A; color: #D0D0D0; font-size: 12px;")

            # Обновляем текст пункта меню
            self.toggle_theme_action.setText("Светлая тема")

        else:
            # Светлая тема
            self.setStyleSheet("background-color: #F0F0F0; color: #333333;")

            # Основные элементы интерфейса
            self.scroll_area.setStyleSheet("background-color: #E0E0E0;")
            self.drawing_area.setStyleSheet("background-color: #E0E0E0;")

            # Панель истории
            if hasattr(self, 'history_group'):
                self.history_group.setStyleSheet("""
                    QGroupBox {
                        background-color: #ECECEC;
                        color: #333333;
                        border: 2px solid #D0D0D0;
                        border-radius: 8px;
                        margin-top: 8px;
                        font-size: 12px;
                        font-weight: normal;
                    }
                """)

            if hasattr(self, 'history_list'):
                self.history_list.setStyleSheet("""
                    QListWidget {
                        background-color: #ECECEC; 
                        color: #333333; 
                        font-size: 12px; 
                        border: none;
                        padding: 2px;
                    }
                    QListWidget::item {
                        padding: 4px 2px;
                        border-bottom: 1px solid #D0D0D0;
                    }
                    QListWidget::item:selected {
                        background-color: #C0C0C0;
                    }
                    QScrollBar:vertical {
                        border: none;
                        background: #E0E0E0;
                        width: 8px;
                        margin: 0px;
                    }
                    QScrollBar::handle:vertical {
                        background: #A0A0A0;
                        min-height: 20px;
                        border-radius: 4px;
                    }
                    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                        height: 0px;
                    }
                """)

            # Заголовок истории действий
            if hasattr(self, 'history_title'):
                self.history_title.setStyleSheet("color: #333333; font-size: 12px; font-weight: bold;")  # Темный текст для светлой темы
            
            # Кнопка закрытия истории
            if hasattr(self, 'close_history_button'):
                self.close_history_button.setStyleSheet("""
                    QPushButton {
                        background-color: transparent;
                        color: #666666;
                        border: none;
                        font-size: 10px;
                        font-weight: bold;
                    }
                    QPushButton:hover {
                        color: #333333;
                    }
                    QPushButton:pressed {
                        color: #999999;
                    }
                """)

            # Гамма-коррекция
            if hasattr(self, 'gamma_group'):
                self.gamma_group.setStyleSheet("background-color: #ECECEC; color: #333333;")

            if hasattr(self, 'gamma_label'):
                self.gamma_label.setStyleSheet("color: #333333; font-size: 12px;")

            if hasattr(self, 'gamma_slider'):
                self.gamma_slider.setStyleSheet("""
                    QSlider::groove:horizontal {
                        background: #C0C0C0;
                        height: 8px;
                        border-radius: 4px;
                    }
                    QSlider::handle:horizontal {
                        background: #666666;
                        width: 16px;
                        margin: -4px 0;
                        border-radius: 8px;
                    }
                """)

            if hasattr(self, 'reset_gamma_button') and hasattr(self, 'apply_gamma_button'):
                self.reset_gamma_button.setStyleSheet("background-color: #D0D0D0; color: #333333;")
                self.apply_gamma_button.setStyleSheet("background-color: #D0D0D0; color: #333333;")

            # Гистограмма
            if hasattr(self, 'histogram_group'):
                self.histogram_group.setStyleSheet("""
                    QGroupBox {
                        background-color: #ECECEC;
                        color: #333333;
                        border: 2px solid #D0D0D0;
                        border-radius: 8px;
                        margin-top: 8px;
                        font-size: 12px;
                        font-weight: normal;
                        padding-top: 12px;
                    }
                    QGroupBox:title {
                        subcontrol-origin: margin;
                        left: 10px;
                        top: 2px;
                        padding: 0 3px;
                    }
                """)

            if hasattr(self, 'histogram_label'):
                self.histogram_label.setStyleSheet("background-color: #F5F5F5; border: none;")

            # Кнопки отмены/повтора
            if hasattr(self, 'undo_button') and hasattr(self, 'redo_button'):
                button_style = """
                    QPushButton {
                        background-color: #ECECEC;
                        color: #333333;
                        border: 2px solid #D0D0D0;
                        border-radius: 8px;
                        padding: 2px;
                    }
                    QPushButton:hover {
                        background-color: #D0D0D0;
                        border-color: #A0A0A0;
                    }
                    QPushButton:pressed {
                        background-color: #C0C0C0;
                    }
                """
                self.undo_button.setStyleSheet(button_style)
                self.redo_button.setStyleSheet(button_style)

            # Меню
            self.menuBar().setStyleSheet("""
                QMenuBar {
                    background-color: #E0E0E0;
                    color: #333333;
                    font-family: 'Roboto';
                    font-size: 9pt;
                }
                QMenuBar::item {
                    background-color: #E0E0E0;
                    color: #333333;
                    padding: 5px 15px;
                }
                QMenuBar::item:selected {
                    background-color: #C0C0C0;
                    color: #333333;
                }
                QMenu::item {
                    background-color: #E0E0E0;
                    color: #333333;
                    padding: 5px 10px;
                }
                QMenu::item:hover {
                    background-color: #C0C0C0;
                    color: #333333;
                }
                QMenu::item:selected {
                    background-color: #C0C0C0;
                    color: #333333;
                }
            """)

            # Статусбар
            if self.statusBar():
                self.statusBar().setStyleSheet("background-color: #E0E0E0; color: #333333; font-size: 12px;")

            # Обновляем текст пункта меню
            self.toggle_theme_action.setText("Темная тема")


if __name__ == "__main__":
    app = QApplication([])
    window = GraphicEditor()
    window.show()
    app.exec()