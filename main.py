import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import *
import cv2
import os
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision
import torch.nn.functional as F
from torchsummary import summary
from PIL import Image
import matplotlib.pyplot as plt

imgs = []
imgs_PIL = []
filenames = []
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
        global imgs, imgs_PIL, filenames
        imgs = []  # clear the list
        imgs_PIL = []  # clear the list
        filenames = []  # clear the list
        for filename in os.listdir(folder_path):
            try:
                img_PIL = Image.open(os.path.join(folder_path, filename))
                img = cv2.imread(os.path.join(folder_path, filename))

                if img is not None:
                    imgs.append(img)
                if img_PIL is not None:
                    imgs_PIL.append(img_PIL)

                filenames.append(filename)
            except:
                pass

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
        self.chessboard_size = (11, 8)
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

        chessboard_size = self.chessboard_size  # chessboard size
        win_size = self.win_size  # window size
        zero_zone = self.win_size  # this parameter means to ignore
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
        chessboard_size = self.chessboard_size  # chessboard size
        win_size = self.win_size  # window size
        zero_zone = self.win_size  # this parameter means to ignore
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)  # criteria
        for img in imgs:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # grayscale the image
            ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
            if ret:
                corners = cv2.cornerSubPix(gray, corners, win_size, zero_zone, criteria)
                self.image_points.append(corners)

                # prepare object points
                object_point = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
                object_point[:, :2] = np.mgrid[0 : chessboard_size[0], 0 : chessboard_size[1]].T.reshape(-1, 2)
                self.object_points.append(object_point)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.object_points, self.image_points, imgs[0].shape[:2], None, None)
        print("Intrinsic Matrix: ")
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
        extrinsic_matrix = np.column_stack((cv2.Rodrigues(rvecs)[0], tvecs))
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

        # initialize parameters
        self.params()

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
        line_edit.textChanged.connect(self.on_line_edit_changed)
        # show words on board button
        show_words_on_board_button = QPushButton("2.1 Show words on board")
        show_words_on_board_button.clicked.connect(self.show_words_on_board)
        # show words vertical button
        show_words_vertical_button = QPushButton("2.2 Show words vertical")
        show_words_vertical_button.clicked.connect(self.show_words_vertical)

        # add title label to layout
        self.layout.addWidget(title_label)
        # add other objects to layout
        self.layout.addWidget(line_edit)
        self.layout.addWidget(show_words_on_board_button)
        self.layout.addWidget(show_words_vertical_button)

    def params(self):
        self.chessboard_size = (11, 8)
        self.win_size = (5, 5)
        self.zero_zone = (-1, -1)
        self.text = ""
        self.object_points = []
        self.image_points = []
        self.intrinsic_matrix = None
        self.distortion_coefficients = None
        self.rvecs = None
        self.tvecs = None
        self.letter_3d_coordinates = {}

    def on_line_edit_changed(self, text):
        print("Line edit text changed:", text)
        self.text = text

    def find_intrinsic(self):
        global imgs
        self.object_points = []  # 3D points in real world space
        self.image_points = []  # 2D points in image plane
        chessboard_size = self.chessboard_size  # chessboard size
        win_size = self.win_size  # window size
        zero_zone = self.win_size  # this parameter means to ignore
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)  # criteria
        for img in imgs:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # grayscale the image
            ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
            if ret:
                corners = cv2.cornerSubPix(gray, corners, win_size, zero_zone, criteria)
                self.image_points.append(corners)

                # prepare object points
                object_point = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
                object_point[:, :2] = np.mgrid[0 : chessboard_size[0], 0 : chessboard_size[1]].T.reshape(-1, 2)
                self.object_points.append(object_point)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.object_points, self.image_points, imgs[0].shape[:2], None, None)
        print("Intrinsic Matrix: ")
        print(mtx)
        self.intrinsic_matrix, self.distortion_coefficients, self.rvecs, self.tvecs = mtx, dist, rvecs, tvecs

    def load_onboard_txt(self):
        self.letter_3d_coordinates = {}  # clear the dictionary

        # select the txt file
        # file_path = QFileDialog.getOpenFileName(self, "Select File")
        file_path = "./alphabet_lib_onboard.txt"  # fix the path

        fs = cv2.FileStorage(file_path, cv2.FILE_STORAGE_READ)  # open the file for reading

        # read the letters
        for letter in self.text:
            # read the letter
            letter_3d_coordinate = fs.getNode(letter).mat().astype(np.float32)
            # reshape the letter
            letter_3d_coordinate = letter_3d_coordinate.reshape(-1, 3)
            # store the letter
            self.letter_3d_coordinates[letter] = letter_3d_coordinate

    def load_vertical_txt(self):
        self.letter_3d_coordinates = {}  # clear the dictionary

        # select the txt file
        # file_path = QFileDialog.getOpenFileName(self, "Select File")
        file_path = "./alphabet_lib_vertical.txt"  # fix the path

        fs = cv2.FileStorage(file_path, cv2.FILE_STORAGE_READ)  # open the file for reading

        # read the letters
        for letter in self.text:
            # read the letter
            letter_3d_coordinate = fs.getNode(letter).mat().astype(np.float32)
            # reshape the letter
            letter_3d_coordinate = letter_3d_coordinate.reshape(-1, 3)
            # store the letter
            self.letter_3d_coordinates[letter] = letter_3d_coordinate

    def offset_letter(self, index, letter):
        letter_3d_coordinate = self.letter_3d_coordinates[letter].copy()
        # define the 6 block for the chessboard
        offset = {
            0: [7, 5, 0],
            1: [4, 5, 0],
            2: [1, 5, 0],
            3: [7, 2, 0],
            4: [4, 2, 0],
            5: [1, 2, 0],
        }
        # offset the letter
        for i, coordinate in enumerate(letter_3d_coordinate):
            coordinate[0] += offset[index][0]
            coordinate[1] += offset[index][1]
            coordinate[2] += offset[index][2]
            letter_3d_coordinate[i] = coordinate

        # print(letter_3d_coordinate)

        return letter_3d_coordinate

    def show_words_on_board(self):
        global imgs
        if len(imgs) == 0:
            print("Please load images first")
            return

        if len(self.text) == 0:
            print("Please input text first")
            return

        self.find_intrinsic()  # get the intrinsic(for all imgs)
        self.load_onboard_txt()

        # global imgs
        for index, img in enumerate(imgs):
            img = img.copy()
            rvec = self.rvecs[index]
            tvec = self.tvecs[index]
            for text_index, letter in enumerate(self.text):
                letter_3d_coordinate = self.offset_letter(text_index, letter)

                # project the 3D letter to 2D
                image_points, _ = cv2.projectPoints(letter_3d_coordinate, rvec, tvec, self.intrinsic_matrix, self.distortion_coefficients)
                # print(letter)
                # print(self.letter_3d_coordinates[letter])
                # print(image_points.shape)

                # draw the letter
                for i in range(0, len(image_points), 2):
                    pt1 = (int(image_points[i][0][0]), int(image_points[i][0][1]))
                    pt2 = (int(image_points[i + 1][0][0]), int(image_points[i + 1][0][1]))
                    cv2.line(img, pt1, pt2, (0, 0, 255), 5)

            # show the image
            cv2.imshow("img", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def show_words_vertical(self):
        global imgs
        if len(imgs) == 0:
            print("Please load images first")
            return

        if len(self.text) == 0:
            print("Please input text first")
            return

        self.find_intrinsic()  # get the intrinsic(for all imgs)
        self.load_vertical_txt()

        # global imgs
        for index, img in enumerate(imgs):
            img = img.copy()
            rvec = self.rvecs[index]
            tvec = self.tvecs[index]
            for text_index, letter in enumerate(self.text):
                letter_3d_coordinate = self.offset_letter(text_index, letter)

                # project the 3D letter to 2D
                image_points, _ = cv2.projectPoints(letter_3d_coordinate, rvec, tvec, self.intrinsic_matrix, self.distortion_coefficients)
                # print(letter)
                # print(self.letter_3d_coordinates[letter])
                # print(image_points.shape)

                # draw the letter
                for i in range(0, len(image_points), 2):
                    pt1 = (int(image_points[i][0][0]), int(image_points[i][0][1]))
                    pt2 = (int(image_points[i + 1][0][0]), int(image_points[i + 1][0][1]))
                    cv2.line(img, pt1, pt2, (0, 0, 255), 5)

            # show the image
            cv2.imshow("img", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


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
        block_size = 25  # must be odd and between 5 and 50
        stereo = cv2.StereoBM_create(numDisparities=num_disparities, blockSize=block_size)

        # compute the disparity map
        disparity_map = stereo.compute(image_L_gray, image_R_gray)

        # normalize the disparity map
        min_disparity = disparity_map.min()
        max_disparity = disparity_map.max()
        self.disparity_map_normalized = ((disparity_map - min_disparity) / (max_disparity - min_disparity) * 255).astype(np.uint8)

        # show the disparity map
        cv2.imshow("disparity map", self.disparity_map_normalized)

        # show image L and R
        self.show_image_L_and_R()

        # set the mouse callback function
        cv2.setMouseCallback("image_L", self.check_disparity_value)

    def show_image_L_and_R(self):
        global image_L, image_R
        if image_L is None or image_R is None:
            print("Please load images first")
            return

        # show the images
        cv2.imshow("image_L", image_L)
        cv2.imshow("image_R", image_R)

    def check_disparity_value(self, event, x, y, flags, param):
        global image_L, image_R
        if event == cv2.EVENT_LBUTTONDOWN:
            # get the depth
            disparity_value = self.disparity_map_normalized[y, x]
            # compute the corresponding point
            corresponding_x = x - disparity_value

            print(f"({x}, y={y}), dis={disparity_value}")

            # mark the corresponding point at image_R
            cv2.circle(image_R, (corresponding_x, y), 15, (0, 255, 0), -1)
            cv2.imshow("image_R", image_R)

        if event == cv2.EVENT_RBUTTONDOWN:
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
        load_image1_button.clicked.connect(self.load_image1)
        # load image2 button
        load_image2_button = QPushButton("4.2 Load Image2")
        load_image2_button.clicked.connect(self.load_image2)
        # keypoints button
        keypoints_button = QPushButton("4.3 Keypoints")
        keypoints_button.clicked.connect(self.keypoints)
        # matched keypoints button
        matched_keypoints_button = QPushButton("4.4 Matched keypoints")
        matched_keypoints_button.clicked.connect(self.matched_keypoints)

        # add title label to layout
        self.layout.addWidget(title_label)
        # add other objects to layout
        self.layout.addWidget(load_image1_button)
        self.layout.addWidget(load_image2_button)
        self.layout.addWidget(keypoints_button)
        self.layout.addWidget(matched_keypoints_button)

    def params(self):
        self.image1 = None
        self.image2 = None

    def load_image1(self):
        # open a dialog to select a file
        file_path = QFileDialog.getOpenFileName(self, "Select File")
        # get the image
        self.image1 = cv2.imread(file_path[0])

    def load_image2(self):
        # open a dialog to select a file
        file_path = QFileDialog.getOpenFileName(self, "Select File")
        # get the image
        self.image2 = cv2.imread(file_path[0])

    def keypoints(self):
        # convert to grayscale
        image_gray = cv2.cvtColor(self.image1, cv2.COLOR_BGR2GRAY)

        # create a SIFT object
        sift = cv2.SIFT_create()

        # detect keypoints and compute descriptors
        keypoints, descriptors = sift.detectAndCompute(image_gray, None)

        # draw keypoints
        image_with_keypoints = cv2.drawKeypoints(image_gray, keypoints, None, color=(0, 255, 0))

        # show the images
        cv2.imshow("image_L_with_keypoints", image_with_keypoints)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def matched_keypoints(self):
        # convert to grayscale
        image1_gray = cv2.cvtColor(self.image1, cv2.COLOR_BGR2GRAY)
        image2_gray = cv2.cvtColor(self.image2, cv2.COLOR_BGR2GRAY)

        # create a SIFT object
        sift = cv2.SIFT_create()

        # detect keypoints and compute descriptors
        keypoints1, descriptors1 = sift.detectAndCompute(image1_gray, None)
        keypoints2, descriptors2 = sift.detectAndCompute(image2_gray, None)

        # create a BFMatcher object
        bf = cv2.BFMatcher()

        # match descriptors
        matches = bf.knnMatch(descriptors1, descriptors2, k=2)

        # apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append([m])

        # draw matches
        image_with_matches = cv2.drawMatchesKnn(
            image1_gray, keypoints1, image2_gray, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        # show the images
        cv2.imshow("image_with_matches", image_with_matches)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


class VGG19(QFrame):
    def __init__(self):
        super().__init__()

        # initialize parameters
        self.params()
        self.load_model()

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
        load_image_button.clicked.connect(self.load_image)
        # show augmented images button
        show_augmented_images_button = QPushButton("5.1 Show Augmented Images")
        show_augmented_images_button.clicked.connect(self.show_augmented_images)
        # show model structure button
        show_model_structure_button = QPushButton("5.2 Show Model Structure")
        show_model_structure_button.clicked.connect(self.show_model_structure)
        # show Acc and Loss button
        show_acc_and_loss_button = QPushButton("5.3 Show Acc and Loss")
        show_acc_and_loss_button.clicked.connect(self.show_acc_and_loss)
        # inference button
        inference_button = QPushButton("5.4 Inference")
        inference_button.clicked.connect(self.inference)
        # predict label
        self.predict_label = QLabel("Predict= ")
        # graphics view with default text in it
        self.graphics_view = QGraphicsView()
        self.graphics_view.setScene(QGraphicsScene())
        self.graphics_view.scene().addText("Inference Image")
        self.graphics_view.setFixedSize(150, 150)

        # add title label to layout
        self.layout.addWidget(title_label)
        # add other objects to layout
        self.layout.addWidget(load_image_button)
        self.layout.addWidget(show_augmented_images_button)
        self.layout.addWidget(show_model_structure_button)
        self.layout.addWidget(show_acc_and_loss_button)
        self.layout.addWidget(inference_button)
        self.layout.addWidget(self.predict_label)
        self.layout.addWidget(self.graphics_view)

    def params(self):
        self.state_dict = None
        self.model = torchvision.models.vgg19_bn(num_classes=10)
        self.inference_img = None

    def load_model(self):
        # load the model by pth file
        if torch.cuda.is_available():
            self.state_dict = torch.load("model.pth")
        else:
            self.state_dict = torch.load("model.pth", map_location=torch.device("cpu"))
        self.model.load_state_dict(self.state_dict)
        self.model.eval()

    def load_image(self):
        # open a dialog to select a file
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File")
        # get the image
        self.inference_img = Image.open(file_path)
        # show the image in the graphics view and set the size 128x128
        pixmap = QPixmap(file_path)
        scaled_pixmap = pixmap.scaled(128, 128, Qt.KeepAspectRatio)
        self.graphics_view.scene().clear()
        self.graphics_view.setFixedSize(150, 150)
        self.graphics_view.scene().addPixmap(scaled_pixmap)

    def show_augmented_images(self):
        global imgs, imgs_PIL, filenames
        if len(imgs) == 0:
            print("Please load images first")
            return

        # Create a list to store the augmented images
        augmented_images = []

        # Define the data augmentation transformations
        data_transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(30),
            ]
        )

        # Apply the transformations to the images
        for img_PIL in imgs_PIL:
            augmented_images.append(data_transforms(img_PIL))

        # Get the labels from the filenames, and remove the file extensions
        labels = [os.path.splitext(filename)[0] for filename in filenames]

        # Create a new window to display augmented images with labels
        plt.figure(figsize=(10, 10))
        # Show the augmented images
        for index, (img, label) in enumerate(zip(augmented_images, labels)):
            plt.subplot(3, 3, index + 1)
            plt.imshow(img)
            plt.title(label)
        # Show the plot
        plt.suptitle("Augmented Images")
        plt.tight_layout()
        plt.show()

    def show_model_structure(self):
        # Show the summary of the model
        summary(self.model, (3, 32, 32))

    def show_acc_and_loss(self):
        # Load the image
        img = Image.open("./epoch_40.png")
        # Show the image
        plt.figure(figsize=(10, 5))
        plt.imshow(img)
        plt.show()

    def preprocess_image(self, image):
        transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        image = transform(image)
        return image

    def inference(self):
        img = self.preprocess_image(self.inference_img)

        # inference
        with torch.no_grad():
            output = self.model(img.unsqueeze(0))

        # labels
        labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        print(output.data)
        probabilities = F.softmax(output.data, dim=1).squeeze()
        print(probabilities)
        # Show the probability distribution of model prediction in new window.
        plt.figure(figsize=(10, 5))
        plt.bar(labels, probabilities)
        plt.title("Probability of each class")
        plt.xlabel("Class")
        plt.ylabel("Probability")
        plt.show()
        # Set the predicted label
        predicted_label = labels[torch.argmax(probabilities)]
        # Show the predicted label in the label
        self.predict_label.setText("Predict= " + predicted_label)


if __name__ == "__main__":
    app = QApplication(sys.argv)  # initialize the application

    mainWindow = MainWindow()  # create a new instance of the main window
    mainWindow.show()  # make the main window visible

    sys.exit(app.exec_())  # start the event loop
