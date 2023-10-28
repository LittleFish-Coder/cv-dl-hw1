import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import *
import cv2
import os
import numpy as np

imgs = []
image_L = None
image_R = None


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("CV DL HW1")
        self.setGeometry(300, 300, 1000, 600)  # set the position and size of the window

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
        layout1.addWidget(load_image)
        layout1.addWidget(calibration)
        layout1.addWidget(augment_reality)
        layout1.addWidget(stereo_disparity_map)

        # layout 2
        layout2 = QHBoxLayout()
        layout2.addWidget(sift)
        layout2.addWidget(vgg19)

        # set alignment
        layout1.setAlignment(Qt.AlignCenter)
        layout2.setAlignment(Qt.AlignCenter)

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
        # load all the images in the folder
        global imgs
        imgs = []  # clear the list
        for filename in os.listdir(folder_path):
            img = cv2.imread(os.path.join(folder_path, filename))
            if img is not None:
                imgs.append(img)

    def load_image_L(self):
        # open a dialog to select a file
        file_path = QFileDialog.getOpenFileName(self, "Select File")
        # print the path of the selected file
        print(file_path)
        # get the image
        global image_L
        image_L = cv2.imread(file_path[0])

    def load_image_R(self):
        # open a dialog to select a file
        file_path = QFileDialog.getOpenFileName(self, "Select File")
        # print the path of the selected file
        print(file_path)
        # get the image
        global image_R
        image_R = cv2.imread(file_path[0])


class Calibration(QFrame):
    def __init__(self):
        super().__init__()
        # initialize parameters
        self.params()

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
        find_corners_button.clicked.connect(self.find_corners)
        # find intrinsic button
        find_intrinsic_button = QPushButton("1.2 Find intrinsic")
        find_intrinsic_button.clicked.connect(self.find_intrinsic)
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
        spin_box.valueChanged.connect(self.on_spin_box_value_changed)
        find_extrinsic_button = QPushButton("Find extrinsic")  # add button
        find_extrinsic_button.clicked.connect(self.find_extrinsic)
        # add to layout of find extrinsic frame
        find_extrinsic_layout.addWidget(find_extrinsic_label)
        find_extrinsic_layout.addWidget(spin_box)
        find_extrinsic_layout.addWidget(find_extrinsic_button)
        # find distortion button
        find_distortion_button = QPushButton("1.4 Find distortion")
        find_distortion_button.clicked.connect(self.find_distortion)
        # show result button
        show_result_button = QPushButton("1.5 Show result")
        show_result_button.clicked.connect(self.show_result)

        # add title label to layout
        self.layout.addWidget(title_label)
        # add other objects to layout
        self.layout.addWidget(find_corners_button)
        self.layout.addWidget(find_intrinsic_button)
        self.layout.addWidget(find_extrinsic_frame)
        self.layout.addWidget(find_distortion_button)
        self.layout.addWidget(show_result_button)

    def params(self):
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.chessboard_size = (8, 11)
        self.win_size = (5, 5)
        self.zero_zone = (-1, -1)
        self.image_points = []
        self.object_points = []
        self.intrinsic_matrix = None
        self.distortion_coefficients = None
        self.spin_box_value = 1

    def find_corners(self):
        global imgs  # get the images
        gray_imgs = []  # store gray images
        corners_imgs = []  # store images with corners

        for img in imgs:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # grayscale the image
            gray_imgs.append(gray)

        chessboard_size = (8, 11)  # chessboard size
        win_size = (5, 5)  # window size
        zero_zone = (-1, -1)  # this parameter means to ignore
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)  # criteria

        # find corners
        for gray in gray_imgs:
            ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
            if ret:
                corners = cv2.cornerSubPix(gray, corners, win_size, zero_zone, criteria)
                # draw corners
                img_with_corners = cv2.drawChessboardCorners(gray, chessboard_size, corners, ret)
                # store the image
                corners_imgs.append(img_with_corners)
            else:
                print("Can't find corners")

        # show the images
        for img in corners_imgs:
            cv2.imshow("img", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def find_intrinsic(self):
        global imgs
        self.object_points = []  # 3D points in real world space
        self.image_points = []  # 2D points in image plane
        chessboard_size = (8, 11)  # chessboard size
        for img in imgs:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # grayscale the image
            ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
            if ret:
                corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                self.image_points.append(corners)

                # prepare object points
                object_point = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
                object_point[:, :2] = np.mgrid[0 : chessboard_size[0], 0 : chessboard_size[1]].T.reshape(-1, 2)
                self.object_points.append(object_point)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.object_points, self.image_points, imgs[0].shape[:2], None, None)
        print("Camera Matrix: ")
        print(mtx)
        self.intrinsic_matrix, self.distortion_coefficients = mtx, dist

    def on_spin_box_value_changed(self, spin_box_value):
        self.spin_box_value = spin_box_value

    def find_extrinsic(self):
        img_index = self.spin_box_value - 1
        object_points = self.object_points[img_index]
        image_points = self.image_points[img_index]
        # estimate the rotation and translation vectors
        ret, rvecs, tvecs = cv2.solvePnP(object_points, image_points, self.intrinsic_matrix, self.distortion_coefficients)
        # Extrinsic Matrix (rotation and translation)
        extrinsic_matrix = np.hstack((rvecs, tvecs))
        print("Extrinsic Matrix: ")
        print(extrinsic_matrix)

    def find_distortion(self):
        print("Distortion Coefficients: (K1, K2, P1, P2, K3)")
        print(self.distortion_coefficients)

    def show_result(self):
        # Undistorted Image
        global imgs
        undistorted_imgs = []
        for img in imgs:
            undistorted_img = cv2.undistort(img, self.intrinsic_matrix, self.distortion_coefficients)
            undistorted_imgs.append(undistorted_img)

        # show the images
        for img, undistorted_img in zip(imgs, undistorted_imgs):
            # write text on the image
            cv2.putText(img, "Distorted Image", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(undistorted_img, "Undistorted Image", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            # concatenate the images
            concatenated_img = np.concatenate((img, undistorted_img), axis=1)
            # show the image
            cv2.imshow("img", concatenated_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


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
        stereo_disparity_map_button.clicked.connect(self.stereo_disparity_map)

        # add title label to layout
        self.layout.addWidget(title_label)
        # add other objects to layout
        self.layout.addWidget(stereo_disparity_map_button)

    def stereo_disparity_map(self):
        global image_L, image_R
        if image_L is None or image_R is None:
            print("Please load images first")
            return
        else:
            # convert to grayscale
            image_L_gray = cv2.cvtColor(image_L, cv2.COLOR_BGR2GRAY)
            image_R_gray = cv2.cvtColor(image_R, cv2.COLOR_BGR2GRAY)

        # create a StereoBM object with default parameters
        num_disparities = 256  # must be divisible by 16
        block_size = 25  # must be odd and between 5 and 255
        stereo = cv2.StereoBM_create(numDisparities=num_disparities, blockSize=block_size)

        # compute the disparity map
        disparity_map = stereo.compute(image_L_gray, image_R_gray)

        # normalize the disparity map
        min_disparity = disparity_map.min()
        max_disparity = disparity_map.max()
        disparity_map_normalized = ((disparity_map - min_disparity) / (max_disparity - min_disparity) * 255).astype(np.uint8)

        # show the disparity map
        cv2.imshow("disparity map", disparity_map_normalized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


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
