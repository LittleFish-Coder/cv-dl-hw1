import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QPalette, QColor


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("CV DL HW1")
        self.setGeometry(300, 300, 1000, 600)  # set the position and size of the window
        self.setMinimumSize(1200, 800)  # set the minimum size of the window

        load_image = LoadImage()
        calibration = Calibration()
        augment_reality = AugmentReality()
        stereo_disparity_map = StereoDisparityMap()
        sift = SIFT()
        vgg19 = VGG19()

        # main layout
        layout = QVBoxLayout(self)

        # layout 1
        layout1 = QHBoxLayout()
        layout1.addStretch(1)
        layout1.addWidget(load_image)
        layout1.addStretch(1)
        layout1.addWidget(calibration)
        layout1.addStretch(1)
        layout1.addWidget(augment_reality)
        layout1.addStretch(1)
        layout1.addWidget(stereo_disparity_map)
        layout1.addStretch(1)

        # layout 2
        layout2 = QHBoxLayout()
        layout2.addStretch(1)
        layout2.addWidget(sift)
        layout2.addStretch(1)
        layout2.addWidget(vgg19)
        layout2.addStretch(1)
        # set alignment
        # layout1.setAlignment(Qt.AlignTop)
        # layout2.setAlignment(Qt.AlignTop)

        layout.addLayout(layout1)
        layout.addLayout(layout2)


class LoadImage(QFrame):
    def __init__(self):
        super().__init__()

        # set border
        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Raised)

        # vertical layout
        self.layout = QVBoxLayout(self)
        self.layout.setSizeConstraint(QLayout.SetFixedSize)

        # title label
        title_label = QLabel("Load Image")
        # load folder button
        load_folder_button = QPushButton("Load folder")
        load_folder_button.clicked.connect(self.load_folder)
        # load image_L button
        load_image_L_button = QPushButton("Load Image_L")
        load_image_L_button.clicked.connect(self.load_image_L)
        # load image_R button
        load_image_R_button = QPushButton("Load Image_R")
        load_image_R_button.clicked.connect(self.load_image_R)

        # add title label to layout
        self.layout.addWidget(title_label)
        # add push button to layout
        self.layout.addWidget(load_folder_button)
        self.layout.addWidget(load_image_L_button)
        self.layout.addWidget(load_image_R_button)

    def load_folder(self):
        # open a dialog to select a folder
        folder_path = QFileDialog.getExistingDirectory(self, "Select Directory")
        # print the path of the selected folder
        print(folder_path)
        # pass

    def load_image_L(self):
        # open a dialog to select a file
        file_path = QFileDialog.getOpenFileName(self, "Select File")
        # print the path of the selected file
        print(file_path)
        # pass

    def load_image_R(self):
        # open a dialog to select a file
        file_path = QFileDialog.getOpenFileName(self, "Select File")
        # print the path of the selected file
        print(file_path)
        # pass


class Calibration(QFrame):
    def __init__(self):
        super().__init__()

        # set border
        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Raised)

        # vertical layout
        self.layout = QVBoxLayout(self)
        self.layout.setSizeConstraint(QLayout.SetFixedSize)

        # title label
        title_label = QLabel("1. Calibration")
        # find corners button
        find_corners_button = QPushButton("1.1 Find corners")
        # find intrinsic button
        find_intrinsic_button = QPushButton("1.2 Find intrinsic")
        # find extrinsic frame
        find_extrinsic_frame = QFrame()
        find_extrinsic_frame.setFrameShape(QFrame.StyledPanel)
        find_extrinsic_frame.setFrameShadow(QFrame.Raised)
        find_extrinsic_layout = QVBoxLayout(find_extrinsic_frame)
        find_extrinsic_layout.setSizeConstraint(QLayout.SetFixedSize)
        find_extrinsic_label = QLabel("1.3 Find extrinsic")  # add label
        spin_box = QSpinBox()  # add spin box
        spin_box.setRange(1, 15)
        spin_box.setValue(1)
        find_extrinsic_button = QPushButton("Find extrinsic")  # add button
        # add to layout of find extrinsic frame
        find_extrinsic_layout.addWidget(find_extrinsic_label)
        find_extrinsic_layout.addWidget(spin_box)
        find_extrinsic_layout.addWidget(find_extrinsic_button)
        # find distortion button
        find_distortion_button = QPushButton("1.4 Find distortion")
        # show result button
        show_result_button = QPushButton("1.5 Show result")

        # add title label to layout
        self.layout.addWidget(title_label)
        # add other objects to layout
        self.layout.addWidget(find_corners_button)
        self.layout.addWidget(find_intrinsic_button)
        self.layout.addWidget(find_extrinsic_frame)
        self.layout.addWidget(find_distortion_button)
        self.layout.addWidget(show_result_button)


class AugmentReality(QFrame):
    def __init__(self):
        super().__init__()

        # set border
        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Raised)

        # vertical layout
        self.layout = QVBoxLayout(self)
        self.layout.setSizeConstraint(QLayout.SetFixedSize)

        # title label
        title_label = QLabel("2. Augment Reality")
        # line edit
        line_edit = QLineEdit()
        # show words on board button
        show_words_on_board_button = QPushButton("2.1 Show words on board")
        # show words vertical button
        show_words_vertical_button = QPushButton("2.2 Show words vertical")

        # add title label to layout
        self.layout.addWidget(title_label)
        # add other objects to layout
        self.layout.addWidget(line_edit)
        self.layout.addWidget(show_words_on_board_button)
        self.layout.addWidget(show_words_vertical_button)


class StereoDisparityMap(QFrame):
    def __init__(self):
        super().__init__()

        # set border
        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Raised)

        # vertical layout
        self.layout = QVBoxLayout(self)
        self.layout.setSizeConstraint(QLayout.SetFixedSize)

        # title label
        title_label = QLabel("3. Stereo Disparity Map")
        # stereo disparity map button
        stereo_disparity_map_button = QPushButton("3.1 Stereo disparity map")

        # add title label to layout
        self.layout.addWidget(title_label)
        # add other objects to layout
        self.layout.addWidget(stereo_disparity_map_button)


class SIFT(QFrame):
    def __init__(self):
        super().__init__()

        # set border
        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Raised)

        # vertical layout
        self.layout = QVBoxLayout(self)
        self.layout.setSizeConstraint(QLayout.SetFixedSize)

        # title label
        title_label = QLabel("4. SIFT")
        # load image1 button
        load_image1_button = QPushButton("4.1 Load Image1")
        # load image2 button
        load_image2_button = QPushButton("4.2 Load Image2")
        # keypoints button
        keypoints_button = QPushButton("4.3 Keypoints")
        # matched keypoints button
        matched_keypoints_button = QPushButton("4.4 Matched keypoints")

        # add title label to layout
        self.layout.addWidget(title_label)
        # add other objects to layout
        self.layout.addWidget(load_image1_button)
        self.layout.addWidget(load_image2_button)
        self.layout.addWidget(keypoints_button)
        self.layout.addWidget(matched_keypoints_button)


class VGG19(QFrame):
    def __init__(self):
        super().__init__()

        # set border
        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Raised)

        # vertical layout
        self.layout = QVBoxLayout(self)
        self.layout.setSizeConstraint(QLayout.SetFixedSize)

        # title label
        title_label = QLabel("5. VGG19")
        # load image button
        load_image_button = QPushButton("Load Image")
        # show augmented images button
        show_augmented_images_button = QPushButton("5.1 Show Augmented Images")
        # show model structure button
        show_model_structure_button = QPushButton("5.2 Show Model Structure")
        # show Acc and Loss button
        show_acc_and_loss_button = QPushButton("5.3 Show Acc and Loss")
        # inference button
        inference_button = QPushButton("5.4 Inference")
        # predict label
        predict_label = QLabel("Predict= ")
        # graphics view with default text in it
        graphics_view = QGraphicsView()
        graphics_view.setScene(QGraphicsScene())
        graphics_view.scene().addText("Inference Image")

        # add title label to layout
        self.layout.addWidget(title_label)
        # add other objects to layout
        self.layout.addWidget(load_image_button)
        self.layout.addWidget(show_augmented_images_button)
        self.layout.addWidget(show_model_structure_button)
        self.layout.addWidget(show_acc_and_loss_button)
        self.layout.addWidget(inference_button)
        self.layout.addWidget(predict_label)
        self.layout.addWidget(graphics_view)


if __name__ == "__main__":
    app = QApplication(sys.argv)  # initialize the application

    mainWindow = MainWindow()  # create a new instance of the main window
    mainWindow.show()  # make the main window visible

    sys.exit(app.exec_())  # start the event loop
