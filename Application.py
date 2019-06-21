
from PyQt5 import QtCore, QtGui, QtWidgets
import images
from PyQt5 import QtCore, QtGui, QtWidgets
import sys

from PyQt5.QtWidgets import QApplication, QDialog
from PyQt5.uic import loadUi
import imutils
import numpy as np
from sklearn.metrics import pairwise
from PyQt5 import QtCore


bg = None
bg2 = None


def Backend(self):
    import cv2
    import imutils
    import numpy as np
    from sklearn.metrics import pairwise

    bg = None
    bg2 = None

    def run_avg(image, aWeight):
        global bg
        if bg is None:
            bg = image.copy().astype("float")
            return
        cv2.accumulateWeighted(image, bg, aWeight)

    def run_avg2(image, aWeight):
        global bg2
        if bg2 is None:
            bg2 = image.copy().astype('float')
            return
        cv2.accumulateWeighted(image, bg2, aWeight)

    def segment(image, threshold=25):
        global bg
        diff = cv2.absdiff(bg.astype("uint8"), image)
        thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
        (_, cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(cnts) == 0:
            return
        else:
            segmented = max(cnts, key=cv2.contourArea)
            return (thresholded, segmented)

    def segment2(image, threshold=25):
        global bg2
        diff = cv2.absdiff(bg2.astype("uint8"), image)
        thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
        (_, cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(cnts) == 0:
            return
        else:
            segmented = max(cnts, key=cv2.contourArea)
            return (thresholded, segmented)

    def count(thresholded, segmented):
        chull = cv2.convexHull(segmented)
        extreme_top = tuple(chull[chull[:, :, 1].argmin()][0])
        extreme_bottom = tuple(chull[chull[:, :, 1].argmax()][0])
        extreme_left = tuple(chull[chull[:, :, 0].argmin()][0])
        extreme_right = tuple(chull[chull[:, :, 0].argmax()][0])
        cX = (extreme_left[0] + extreme_right[0]) / 2
        cX = int(cX)
        cY = (extreme_top[0] + extreme_bottom[0]) / 2
        cY = int(cY)
        distance = \
        pairwise.euclidean_distances([(cX, cY)], Y=[extreme_left, extreme_right, extreme_top, extreme_bottom])[0]
        maximum_distance = distance[distance.argmax()]
        radius = int(0.8 * maximum_distance)
        circumference = (2 * np.pi * radius)
        circular_roi = np.zeros(thresholded.shape[:2], dtype="uint8")
        cv2.circle(circular_roi, (cX, cY), radius, 255, 1)
        circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)
        (_, cnts, _) = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        count = 0
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            if ((cY + (cY * 0.25)) > (y + h)) and ((circumference * 0.25) > c.shape[0]):
                count += 1
        return count

    if __name__ == "__main__":

        aWeight = 0.5
        camera = cv2.VideoCapture(0)
        top, right, bottom, left = 10, 10, 370, 370
        top2, right2, bottom2, left2 = 10, 810, 370, 1170
        num_frames = 0
        calibrated = False
        while (True):
            (grabbed, frame) = camera.read()
            frame = imutils.resize(frame, width=1180)
            frame = cv2.flip(frame, 1)
            clone = frame.copy()
            (height, width) = frame.shape[:2]
            roi = frame[top:bottom, right:left]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (7, 7), 0)
            roi2 = frame[top2:bottom2, right2:left2]
            gray2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.GaussianBlur(gray2, (7, 7), 0)
            if num_frames < 30:
                run_avg(gray, aWeight)
                run_avg2(gray2, aWeight)
                if num_frames == 1:
                    print("[STATUS]:Please wait! calibrating...")
                elif num_frames == 29:
                    print("[STATUS]:Calibration successful..")
            else:
                hand = segment(gray)
                hand2 = segment2(gray2)
                fingers, fingers2 = 0, 0
                if hand is not None:
                    (thresholded, segmented) = hand
                    cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                    fingers = count(thresholded, segmented)
                    cv2.putText(clone, str(fingers), (left + 20, top + 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                    cv2.namedWindow("Thresholed1")  # Create a named window
                    cv2.moveWindow("Thresholded1", right + 100, bottom + 120)  # Move it to (40,30)
                    # cv2.waitKey(1)
                    cv2.imshow("Thresholded1", thresholded)

                if hand2 is not None:
                    (thresholded2, segmented2) = hand2
                    cv2.drawContours(clone, [segmented2 + (right2, top2)], -1, (0, 0, 255))
                    fingers2 = count(thresholded2, segmented2)
                    cv2.putText(clone, str(fingers2), (right2 - 70, top2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 2,
                                (0, 0, 255), 3)
                    cv2.namedWindow("Thresholed2")  # Create a named window
                    cv2.moveWindow("Thresholded2", right2 + 100, bottom + 120)  # Move it to (40,30)
                    # cv2.waitKey(1)
                    cv2.imshow("Thresholded2", thresholded2)

                    cv2.putText(clone, "Total: " + str(fingers + fingers2), (left + 140, top2 + 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    cv2.putText(clone, "Concatenation: " + str(fingers) + str(fingers2), (left + 60, top2 + 400),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(clone, (left2, top2), (right2, bottom2), (0, 255, 0), 2)

            num_frames += 1
            cv2.imshow("Video Feed", clone)
            keypress = cv2.waitKey(1) & 0xFF

            if keypress == ord("q"):
                break
    camera.release()
    cv2.destroyAllWindows()


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1390, 835)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/pictures/800px_COLOURBOX17930888.jpg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        MainWindow.setStyleSheet("background-color: rgb(0, 0, 0);\n"
                                 "background-image: url(:/pictures/1280x720-data_out_61_402961734-hand-wallpapers.jpg);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.stackedWidget = QtWidgets.QStackedWidget(self.centralwidget)
        self.stackedWidget.setGeometry(QtCore.QRect(-80, 30, 1421, 841))
        self.stackedWidget.setToolTip("")
        self.stackedWidget.setStyleSheet("")
        self.stackedWidget.setObjectName("stackedWidget")
        self.page1 = QtWidgets.QWidget()
        self.page1.setObjectName("page1")
        self.moreButton = QtWidgets.QPushButton(self.page1)
        self.moreButton.setGeometry(QtCore.QRect(1060, 270, 311, 141))
        self.moreButton.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.moreButton.setStyleSheet("background-color: rgb(0, 30, 39);\n"
                                      "font: 28pt \"MS Shell Dlg 2\";\n"
                                      "selection-background-color: rgb(4, 149, 157);\n"
                                      "color: rgb(255, 255, 255);")
        self.moreButton.setObjectName("moreButton")
        self.closeButton = QtWidgets.QPushButton(self.page1)
        self.closeButton.setGeometry(QtCore.QRect(1060, 450, 311, 141))
        self.closeButton.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.closeButton.setAutoFillBackground(False)
        self.closeButton.setStyleSheet("background-color: rgb(0, 30, 39);\n"
                                       "font: 28pt \"MS Shell Dlg 2\";\n"
                                       "selection-background-color: rgb(4, 149, 157);\n"
                                       "color: rgb(255, 255, 255);")
        self.closeButton.setObjectName("closeButton")
        self.label = QtWidgets.QLabel(self.page1)
        self.label.setGeometry(QtCore.QRect(100, -10, 861, 201))
        self.label.setStyleSheet("font: 48pt \"Wide Latin\";\n"
                                 "color: rgb(255, 255, 255);")
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.page1)
        self.label_2.setGeometry(QtCore.QRect(1040, 250, 47, 13))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.page1)
        self.label_3.setGeometry(QtCore.QRect(1060, 410, 47, 13))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.page1)
        self.label_4.setGeometry(QtCore.QRect(1050, 590, 47, 13))
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.page1)
        self.label_5.setGeometry(QtCore.QRect(70, 550, 991, 121))
        self.label_5.setStyleSheet("font: 48pt \"Wide Latin\";\n"
                                   "color: rgb(85, 255, 127);")
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.appButton = QtWidgets.QPushButton(self.page1)
        self.appButton.setGeometry(QtCore.QRect(1060, 90, 311, 141))
        self.appButton.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.appButton.setStyleSheet("\n"
                                     "background-color: rgb(0, 30, 39);\n"
                                     "font: 28pt \"MS Shell Dlg 2\";\n"
                                     "selection-background-color: rgb(4, 149, 157);\n"
                                     "color: rgb(255, 255, 255);")
        self.appButton.setObjectName("appButton")
        self.label_7 = QtWidgets.QLabel(self.page1)
        self.label_7.setGeometry(QtCore.QRect(90, 190, 941, 351))
        self.label_7.setText("")
        self.label_7.setPixmap(QtGui.QPixmap(":/pictures/maxresdefault.jpg"))
        self.label_7.setScaledContents(True)
        self.label_7.setObjectName("label_7")
        self.stackedWidget.addWidget(self.page1)
        self.page2 = QtWidgets.QWidget()
        self.page2.setObjectName("page2")
        self.label_11 = QtWidgets.QLabel(self.page2)
        self.label_11.setGeometry(QtCore.QRect(520, 280, 47, 13))
        self.label_11.setObjectName("label_11")
        self.label_12 = QtWidgets.QLabel(self.page2)
        self.label_12.setGeometry(QtCore.QRect(970, 740, 47, 13))
        self.label_12.setObjectName("label_12")
        self.startprojectButton = QtWidgets.QPushButton(self.page2)
        self.startprojectButton.setGeometry(QtCore.QRect(120, 0, 1211, 221))
        self.startprojectButton.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.startprojectButton.setAcceptDrops(False)
        self.startprojectButton.setStyleSheet("\n"
                                              "background-color: rgb(0, 30, 39);\n"
                                              "font: 28pt \"MS Shell Dlg 2\";\n"
                                              "selection-background-color: rgb(4, 149, 157);\n"
                                              "color: rgb(255, 255, 255);")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/pictures/images.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.startprojectButton.setIcon(icon1)
        self.startprojectButton.setIconSize(QtCore.QSize(250, 250))
        self.startprojectButton.setAutoDefault(False)
        self.startprojectButton.setObjectName("startprojectButton")
        self.backHomeFromAppButton = QtWidgets.QPushButton(self.page2)
        self.backHomeFromAppButton.setGeometry(QtCore.QRect(1100, 540, 311, 141))
        self.backHomeFromAppButton.setStyleSheet("background-color: rgb(255, 255, 0);\n"
                                                 "background-color: rgb(0, 30, 39);\n"
                                                 "font: 28pt \"MS Shell Dlg 2\";\n"
                                                 "selection-background-color: rgb(4, 149, 157);\n"
                                                 "color: rgb(255, 255, 255);")
        self.backHomeFromAppButton.setObjectName("backHomeFromAppButton")
        self.label_6 = QtWidgets.QLabel(self.page2)
        self.label_6.setGeometry(QtCore.QRect(90, 230, 781, 481))
        self.label_6.setText("")
        self.label_6.setPixmap(QtGui.QPixmap(":/pictures/282201.jpg"))
        self.label_6.setScaledContents(True)
        self.label_6.setObjectName("label_6")
        self.label_8 = QtWidgets.QLabel(self.page2)
        self.label_8.setGeometry(QtCore.QRect(-60, 290, 711, 41))
        font = QtGui.QFont()
        font.setPointSize(36)
        font.setBold(True)
        font.setWeight(75)
        self.label_8.setFont(font)
        self.label_8.setStyleSheet("color: rgb(255, 255, 255);")
        self.label_8.setAlignment(QtCore.Qt.AlignCenter)
        self.label_8.setObjectName("label_8")
        self.label_9 = QtWidgets.QLabel(self.page2)
        self.label_9.setGeometry(QtCore.QRect(100, 370, 47, 13))
        self.label_9.setObjectName("label_9")
        self.label_10 = QtWidgets.QLabel(self.page2)
        self.label_10.setGeometry(QtCore.QRect(100, 370, 401, 41))
        self.label_10.setStyleSheet("color: rgb(255, 255, 255);\n"
                                    "\n"
                                    "font: 75 11pt \"MS Shell Dlg 2\";")
        self.label_10.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.label_10.setObjectName("label_10")
        self.label_41 = QtWidgets.QLabel(self.page2)
        self.label_41.setGeometry(QtCore.QRect(100, 410, 291, 31))
        self.label_41.setStyleSheet("color: rgb(255, 255, 255);\n"
                                    "\n"
                                    "font: 75 11pt \"MS Shell Dlg 2\";")
        self.label_41.setObjectName("label_41")
        self.label_42 = QtWidgets.QLabel(self.page2)
        self.label_42.setGeometry(QtCore.QRect(100, 450, 461, 41))
        self.label_42.setStyleSheet("color: rgb(255, 255, 255);\n"
                                    "\n"
                                    "font: 75 11pt \"MS Shell Dlg 2\";")
        self.label_42.setObjectName("label_42")
        self.label_43 = QtWidgets.QLabel(self.page2)
        self.label_43.setGeometry(QtCore.QRect(100, 520, 451, 51))
        self.label_43.setStyleSheet("color: rgb(255, 255, 255);\n"
                                    "\n"
                                    "font: 75 11pt \"MS Shell Dlg 2\";")
        self.label_43.setObjectName("label_43")
        self.label_44 = QtWidgets.QLabel(self.page2)
        self.label_44.setGeometry(QtCore.QRect(100, 570, 271, 31))
        self.label_44.setStyleSheet("color: rgb(255, 255, 255);\n"
                                    "\n"
                                    "font: 75 11pt \"MS Shell Dlg 2\";")
        self.label_44.setObjectName("label_44")
        self.label_45 = QtWidgets.QLabel(self.page2)
        self.label_45.setGeometry(QtCore.QRect(100, 610, 591, 31))
        self.label_45.setStyleSheet("color: rgb(255, 255, 255);\n"
                                    "\n"
                                    "font: 75 11pt \"MS Shell Dlg 2\";")
        self.label_45.setObjectName("label_45")
        self.label_47 = QtWidgets.QLabel(self.page2)
        self.label_47.setGeometry(QtCore.QRect(100, 640, 851, 181))
        self.label_47.setText("")
        self.label_47.setObjectName("label_47")
        self.label_48 = QtWidgets.QLabel(self.page2)
        self.label_48.setGeometry(QtCore.QRect(560, 240, 921, 301))
        self.label_48.setText("")
        self.label_48.setPixmap(QtGui.QPixmap(":/pictures/800px_COLOURBOX17930888.jpg"))
        self.label_48.setScaledContents(True)
        self.label_48.setObjectName("label_48")
        self.label_49 = QtWidgets.QLabel(self.page2)
        self.label_49.setGeometry(QtCore.QRect(130, 480, 271, 31))
        self.label_49.setStyleSheet("color: rgb(255, 255, 255);\n"
                                    "\n"
                                    "font: 75 11pt \"MS Shell Dlg 2\";")
        self.label_49.setObjectName("label_49")
        self.stackedWidget.addWidget(self.page2)
        self.page3 = QtWidgets.QWidget()
        self.page3.setObjectName("page3")
        self.plainTextEdit = QtWidgets.QPlainTextEdit(self.page3)
        self.plainTextEdit.setGeometry(QtCore.QRect(290, 30, 571, 171))
        self.plainTextEdit.setStyleSheet("color: rgb(255, 255, 255);")
        self.plainTextEdit.setMaximumBlockCount(100)
        self.plainTextEdit.setObjectName("plainTextEdit")
        self.label_13 = QtWidgets.QLabel(self.page3)
        self.label_13.setGeometry(QtCore.QRect(170, 30, 101, 31))
        self.label_13.setStyleSheet("color: rgb(255, 255, 255);\n"
                                    "\n"
                                    "font: 8pt \"MS Shell Dlg 2\";\n"
                                    "font: 75 14pt \"MS Shell Dlg 2\";")
        self.label_13.setAlignment(QtCore.Qt.AlignCenter)
        self.label_13.setObjectName("label_13")
        self.feedbackButton = QtWidgets.QPushButton(self.page3)
        self.feedbackButton.setGeometry(QtCore.QRect(880, 180, 75, 23))
        self.feedbackButton.setStyleSheet("color: rgb(255, 255, 255);\n"
                                          "\n"
                                          "background-color: rgb(255, 248, 30);\n"
                                          "")
        self.feedbackButton.setObjectName("feedbackButton")
        self.horizontalSlider = QtWidgets.QSlider(self.page3)
        self.horizontalSlider.setGeometry(QtCore.QRect(290, 640, 651, 22))
        self.horizontalSlider.setMaximum(10)
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.horizontalSlider.setObjectName("horizontalSlider")
        self.label_15 = QtWidgets.QLabel(self.page3)
        self.label_15.setGeometry(QtCore.QRect(290, 580, 151, 16))
        self.label_15.setStyleSheet("color: rgb(255, 255, 255);\n"
                                    "font: 75 12pt \"MS Shell Dlg 2\";")
        self.label_15.setObjectName("label_15")
        self.label_16 = QtWidgets.QLabel(self.page3)
        self.label_16.setGeometry(QtCore.QRect(280, 670, 21, 16))
        self.label_16.setStyleSheet("color: rgb(255, 255, 255);")
        self.label_16.setAlignment(QtCore.Qt.AlignCenter)
        self.label_16.setObjectName("label_16")
        self.label_17 = QtWidgets.QLabel(self.page3)
        self.label_17.setGeometry(QtCore.QRect(350, 670, 21, 16))
        self.label_17.setStyleSheet("color: rgb(255, 255, 255);")
        self.label_17.setAlignment(QtCore.Qt.AlignCenter)
        self.label_17.setObjectName("label_17")
        self.label_18 = QtWidgets.QLabel(self.page3)
        self.label_18.setGeometry(QtCore.QRect(410, 670, 21, 16))
        self.label_18.setStyleSheet("color: rgb(255, 255, 255);")
        self.label_18.setAlignment(QtCore.Qt.AlignCenter)
        self.label_18.setObjectName("label_18")
        self.label_19 = QtWidgets.QLabel(self.page3)
        self.label_19.setGeometry(QtCore.QRect(480, 670, 21, 16))
        self.label_19.setStyleSheet("color: rgb(255, 255, 255);")
        self.label_19.setAlignment(QtCore.Qt.AlignCenter)
        self.label_19.setObjectName("label_19")
        self.label_20 = QtWidgets.QLabel(self.page3)
        self.label_20.setGeometry(QtCore.QRect(540, 670, 21, 16))
        self.label_20.setStyleSheet("color: rgb(255, 255, 255);")
        self.label_20.setAlignment(QtCore.Qt.AlignCenter)
        self.label_20.setObjectName("label_20")
        self.label_21 = QtWidgets.QLabel(self.page3)
        self.label_21.setGeometry(QtCore.QRect(610, 670, 21, 16))
        self.label_21.setStyleSheet("color: rgb(255, 255, 255);")
        self.label_21.setAlignment(QtCore.Qt.AlignCenter)
        self.label_21.setObjectName("label_21")
        self.label_22 = QtWidgets.QLabel(self.page3)
        self.label_22.setGeometry(QtCore.QRect(670, 670, 21, 16))
        self.label_22.setStyleSheet("color: rgb(255, 255, 255);")
        self.label_22.setAlignment(QtCore.Qt.AlignCenter)
        self.label_22.setObjectName("label_22")
        self.label_23 = QtWidgets.QLabel(self.page3)
        self.label_23.setGeometry(QtCore.QRect(730, 670, 21, 16))
        self.label_23.setStyleSheet("color: rgb(255, 255, 255);")
        self.label_23.setAlignment(QtCore.Qt.AlignCenter)
        self.label_23.setObjectName("label_23")
        self.label_24 = QtWidgets.QLabel(self.page3)
        self.label_24.setGeometry(QtCore.QRect(800, 670, 21, 16))
        self.label_24.setStyleSheet("color: rgb(255, 255, 255);")
        self.label_24.setAlignment(QtCore.Qt.AlignCenter)
        self.label_24.setObjectName("label_24")
        self.label_25 = QtWidgets.QLabel(self.page3)
        self.label_25.setGeometry(QtCore.QRect(860, 670, 21, 16))
        self.label_25.setStyleSheet("color: rgb(255, 255, 255);")
        self.label_25.setAlignment(QtCore.Qt.AlignCenter)
        self.label_25.setObjectName("label_25")
        self.label_26 = QtWidgets.QLabel(self.page3)
        self.label_26.setGeometry(QtCore.QRect(920, 670, 21, 16))
        self.label_26.setStyleSheet("color: rgb(255, 255, 255);")
        self.label_26.setAlignment(QtCore.Qt.AlignCenter)
        self.label_26.setObjectName("label_26")
        self.label_27 = QtWidgets.QLabel(self.page3)
        self.label_27.setGeometry(QtCore.QRect(1110, 120, 121, 161))
        self.label_27.setText("")
        self.label_27.setPixmap(QtGui.QPixmap(":/pictures/if_users-11_984127.png"))
        self.label_27.setScaledContents(True)
        self.label_27.setObjectName("label_27")
        self.label_28 = QtWidgets.QLabel(self.page3)
        self.label_28.setGeometry(QtCore.QRect(1110, 320, 121, 161))
        self.label_28.setText("")
        self.label_28.setPixmap(QtGui.QPixmap(":/pictures/if_users-3_984116.png"))
        self.label_28.setScaledContents(True)
        self.label_28.setObjectName("label_28")
        self.label_29 = QtWidgets.QLabel(self.page3)
        self.label_29.setGeometry(QtCore.QRect(1110, 520, 121, 161))
        self.label_29.setText("")
        self.label_29.setPixmap(QtGui.QPixmap(":/pictures/if_users-2_984102.png"))
        self.label_29.setScaledContents(True)
        self.label_29.setObjectName("label_29")
        self.label_30 = QtWidgets.QLabel(self.page3)
        self.label_30.setGeometry(QtCore.QRect(950, 20, 341, 71))
        self.label_30.setStyleSheet("color: rgb(255, 255, 255);\n"
                                    "font: 24pt \"MS Reference Sans Serif\";")
        self.label_30.setAlignment(QtCore.Qt.AlignCenter)
        self.label_30.setObjectName("label_30")
        self.label_31 = QtWidgets.QLabel(self.page3)
        self.label_31.setGeometry(QtCore.QRect(1260, 110, 171, 31))
        self.label_31.setStyleSheet("color: rgb(255, 255, 255);\n"
                                    "\n"
                                    "font: 75 11pt \"MS Shell Dlg 2\";")
        self.label_31.setObjectName("label_31")
        self.label_32 = QtWidgets.QLabel(self.page3)
        self.label_32.setGeometry(QtCore.QRect(1260, 310, 171, 41))
        self.label_32.setStyleSheet("color: rgb(255, 255, 255);\n"
                                    "font: 75 11pt \"MS Shell Dlg 2\";")
        self.label_32.setObjectName("label_32")
        self.label_33 = QtWidgets.QLabel(self.page3)
        self.label_33.setGeometry(QtCore.QRect(1260, 520, 171, 41))
        self.label_33.setStyleSheet("color: rgb(255, 255, 255);\n"
                                    "font: 75 11pt \"MS Shell Dlg 2\";")
        self.label_33.setObjectName("label_33")
        self.label_14 = QtWidgets.QLabel(self.page3)
        self.label_14.setGeometry(QtCore.QRect(590, 580, 47, 51))
        self.label_14.setStyleSheet("color: rgb(255, 255, 255);")
        self.label_14.setText("")
        self.label_14.setObjectName("label_14")
        self.label_34 = QtWidgets.QLabel(self.page3)
        self.label_34.setGeometry(QtCore.QRect(1260, 150, 151, 31))
        self.label_34.setStyleSheet("color: rgb(255, 255, 255);\n"
                                    "font: 75 11pt \"MS Shell Dlg 2\";")
        self.label_34.setObjectName("label_34")
        self.label_35 = QtWidgets.QLabel(self.page3)
        self.label_35.setGeometry(QtCore.QRect(1260, 190, 151, 31))
        self.label_35.setStyleSheet("color: rgb(255, 255, 255);\n"
                                    "font: 75 11pt \"MS Shell Dlg 2\";")
        self.label_35.setObjectName("label_35")
        self.label_36 = QtWidgets.QLabel(self.page3)
        self.label_36.setGeometry(QtCore.QRect(1260, 360, 151, 31))
        self.label_36.setStyleSheet("color: rgb(255, 255, 255);\n"
                                    "font: 75 11pt \"MS Shell Dlg 2\";")
        self.label_36.setObjectName("label_36")
        self.label_37 = QtWidgets.QLabel(self.page3)
        self.label_37.setGeometry(QtCore.QRect(1260, 400, 151, 31))
        self.label_37.setStyleSheet("color: rgb(255, 255, 255);\n"
                                    "font: 75 11pt \"MS Shell Dlg 2\";")
        self.label_37.setObjectName("label_37")
        self.label_38 = QtWidgets.QLabel(self.page3)
        self.label_38.setGeometry(QtCore.QRect(1260, 570, 151, 31))
        self.label_38.setStyleSheet("color: rgb(255, 255, 255);\n"
                                    "font: 75 11pt \"MS Shell Dlg 2\";")
        self.label_38.setObjectName("label_38")
        self.label_39 = QtWidgets.QLabel(self.page3)
        self.label_39.setGeometry(QtCore.QRect(1260, 610, 151, 31))
        self.label_39.setStyleSheet("color: rgb(255, 255, 255);\n"
                                    "font: 75 11pt \"MS Shell Dlg 2\";")
        self.label_39.setObjectName("label_39")
        self.backHomeFromMoreButton = QtWidgets.QPushButton(self.page3)
        self.backHomeFromMoreButton.setGeometry(QtCore.QRect(110, 610, 111, 81))
        self.backHomeFromMoreButton.setStyleSheet("background-color: rgb(255, 255, 0);\n"
                                                  "background-color: rgb(0, 30, 39);\n"
                                                  "font: 28pt \"MS Shell Dlg 2\";\n"
                                                  "selection-background-color: rgb(4, 149, 157);\n"
                                                  "color: rgb(255, 255, 255);")
        self.backHomeFromMoreButton.setObjectName("backHomeFromMoreButton")
        self.label_40 = QtWidgets.QLabel(self.page3)
        self.label_40.setGeometry(QtCore.QRect(140, 580, 47, 20))
        self.label_40.setStyleSheet("color: rgb(255, 255, 255);")
        self.label_40.setText("")
        self.label_40.setObjectName("label_40")
        self.label_46 = QtWidgets.QLabel(self.page3)
        self.label_46.setGeometry(QtCore.QRect(90, 200, 47, 13))
        self.label_46.setText("")
        self.label_46.setObjectName("label_46")
        self.stackedWidget.addWidget(self.page3)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.stackedWidget.setCurrentIndex(0)

        self.moreButton.clicked.connect(self.MORE)

        self.closeButton.clicked.connect(QtCore.QCoreApplication.instance().quit)

        self.backHomeFromAppButton.clicked.connect(self.backtohome)

        self.appButton.clicked.connect(self.APP)

        self.horizontalSlider.sliderMoved['int'].connect(self.label_14.setNum)

        self.startprojectButton.clicked.connect(Backend)

        self.backHomeFromMoreButton.clicked.connect(self.backtohome)

        self.feedbackButton.clicked.connect(self.plainTextEdit.clear)

        self.closeButton.clicked.connect(self.label_4.clear)

        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "HAND GESTURE APPLICATION"))
        self.moreButton.setToolTip(_translate("MainWindow",
                                              "<html><head/><body><p><span style=\" font-size:6pt; color:#0055ff;\">MEMBERS</span></p></body></html>"))
        self.moreButton.setText(_translate("MainWindow", "EXTRAS"))
        self.closeButton.setToolTip(_translate("MainWindow",
                                               "<html><head/><body><p><span style=\" font-size:6pt; color:#0055ff;\">CLOSE APP</span></p></body></html>"))
        self.closeButton.setText(_translate("MainWindow", "CLOSE"))
        self.label.setText(_translate("MainWindow", "HAND COUNT"))
        self.label_2.setText(_translate("MainWindow", "TextLabel"))
        self.label_3.setText(_translate("MainWindow", "TextLabel"))
        self.label_4.setText(_translate("MainWindow", "TextLabel"))
        self.label_5.setText(_translate("MainWindow", "RECOGNISTION"))
        self.appButton.setToolTip(_translate("MainWindow",
                                             "<html><head/><body><p><span style=\" font-size:6pt; color:#0055ff;\">GO TO PROJECT</span></p></body></html>"))
        self.appButton.setText(_translate("MainWindow", "APPLICATION"))
        self.label_11.setText(_translate("MainWindow", "a"))
        self.label_12.setText(_translate("MainWindow", "TextLabel"))
        self.startprojectButton.setToolTip(_translate("MainWindow",
                                                      "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
                                                      "<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
                                                      "p, li { white-space: pre-wrap; }\n"
                                                      "</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8pt; font-weight:400; font-style:normal;\">\n"
                                                      "<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:6pt; color:#0055ff;\">CLICK ME</span></p></body></html>"))
        self.startprojectButton.setStatusTip(_translate("MainWindow", "CLICK ME"))
        self.startprojectButton.setText(_translate("MainWindow", "START WEB CAM"))
        self.backHomeFromAppButton.setToolTip(_translate("MainWindow",
                                                         "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
                                                         "<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
                                                         "p, li { white-space: pre-wrap; }\n"
                                                         "</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8pt; font-weight:400; font-style:normal;\">\n"
                                                         "<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:6pt; color:#0055ff;\">GO TO HOME</span></p></body></html>"))
        self.backHomeFromAppButton.setText(_translate("MainWindow", "BACK"))
        self.label_8.setText(_translate("MainWindow", "INSTRUCTIONS :"))
        self.label_9.setText(_translate("MainWindow", "TextLabel"))
        self.label_10.setText(_translate("MainWindow",
                                         "<html><head/><body><p><span style=\" font-weight:600;\">-</span><span style=\" font-weight:600; color:#ffff00;\">1</span><span style=\" font-weight:600;\">- </span>Make sure the device is in <span style=\" font-weight:600; color:#55ff00;\">STABLE POSITION </span></p><p><br/></p></body></html>"))
        self.label_41.setText(_translate("MainWindow",
                                         "<html><head/><body><p><span style=\" font-weight:600;\">-</span><span style=\" font-weight:600; color:#ffff00;\">2</span><span style=\" font-weight:600;\">- </span>Click on<span style=\" font-weight:600; color:#55ff00;\"> START WEB CAM </span>button </p></body></html>"))
        self.label_42.setText(_translate("MainWindow",
                                         "<html><head/><body><p><span style=\" font-weight:600;\">-</span><span style=\" font-weight:600; color:#ffff00;\">3</span><span style=\" font-weight:600;\">- </span>Wait till Camera opens and you see two <span style=\" font-weight:600; color:#55ff00;\">GREEN RECTANGLES</span></p></body></html>"))
        self.label_43.setText(_translate("MainWindow",
                                         "<html><head/><body><p><span style=\" font-weight:600;\">-</span><span style=\" font-weight:600; color:#ffff00;\">4</span><span style=\" font-weight:600;\">- </span>Now move slowly your both <span style=\" font-weight:600; color:#55ff00;\">HANDS</span> in these Green Frames</p></body></html>"))
        self.label_44.setText(_translate("MainWindow",
                                         "<html><head/><body><p><span style=\" font-weight:600;\">-</span><span style=\" font-weight:600; color:#ffff00;\">5</span><span style=\" font-weight:600;\">- </span>Make <span style=\" font-weight:600; color:#55ff00;\">NUMBERS</span> by your Hand.</p></body></html>"))
        self.label_45.setText(_translate("MainWindow",
                                         "<html><head/><body><p><span style=\" font-weight:600;\">-</span><span style=\" font-weight:600; color:#ffff00;\">6</span><span style=\" font-weight:600;\">- </span>You\'ll see Count <span style=\" font-weight:600; color:#55ff00;\">ADDITION</span> and <span style=\" font-weight:600; color:#55ff00;\">CONCATINATION </span><span style=\" font-weight:400;\">as output in middle of screen</span></p></body></html>"))
        self.label_49.setText(_translate("MainWindow", "<html><head/><body><p>in top Corners</p></body></html>"))
        self.plainTextEdit.setPlaceholderText(_translate("MainWindow", "GIVE US YOUR FEEDBACK"))
        self.label_13.setText(_translate("MainWindow", "FEEDBACK"))
        self.feedbackButton.setToolTip(_translate("MainWindow",
                                                  "<html><head/><body><p><span style=\" font-size:6pt; color:#0055ff;\">SUBMIT FEEDBACK</span></p></body></html>"))
        self.feedbackButton.setStatusTip(_translate("MainWindow", "PRESS TO SUBMIT "))
        self.feedbackButton.setText(_translate("MainWindow", "SUBMIT"))
        self.label_15.setText(_translate("MainWindow", "RATE OUR PROJECT"))
        self.label_16.setText(_translate("MainWindow", "1"))
        self.label_17.setText(_translate("MainWindow", "1"))
        self.label_18.setText(_translate("MainWindow", "2"))
        self.label_19.setText(_translate("MainWindow", "3"))
        self.label_20.setText(_translate("MainWindow", "4"))
        self.label_21.setText(_translate("MainWindow", "5"))
        self.label_22.setText(_translate("MainWindow", "6"))
        self.label_23.setText(_translate("MainWindow", "7"))
        self.label_24.setText(_translate("MainWindow", "8"))
        self.label_25.setText(_translate("MainWindow", "9"))
        self.label_26.setText(_translate("MainWindow", "10"))
        self.label_30.setText(_translate("MainWindow", "CONTRIBUTED BY :"))
        self.label_31.setText(_translate("MainWindow", "AMARDEEP PALWADE"))
        self.label_32.setText(_translate("MainWindow", "SURBHI NIMJE"))
        self.label_33.setText(_translate("MainWindow", "SAHIL GOGAVE"))
        self.label_34.setText(_translate("MainWindow", "TE 2-BATCH B"))
        self.label_35.setText(_translate("MainWindow", "ROLL NO :305126"))
        self.label_36.setText(_translate("MainWindow", "TE 2-BATCH B"))
        self.label_37.setText(_translate("MainWindow", "ROLL NO :305123"))
        self.label_38.setText(_translate("MainWindow", "TE 2-BATCH A"))
        self.label_39.setText(_translate("MainWindow", "ROLL NO : 305108"))
        self.backHomeFromMoreButton.setToolTip(_translate("MainWindow",
                                                          "<html><head/><body><p><span style=\" font-size:6pt; color:#0055ff;\">GO TO HOME</span></p></body></html>"))
        self.backHomeFromMoreButton.setStatusTip(_translate("MainWindow", "GO BACK TO MAIN PAGE"))
        self.backHomeFromMoreButton.setText(_translate("MainWindow", "BACK"))

    def MORE(self):
        self.stackedWidget.setCurrentIndex(2)

    def backtohome(self):
        self.stackedWidget.setCurrentIndex(0)

    def APP(self):
        self.stackedWidget.setCurrentIndex(1)


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.showMaximized()
    sys.exit(app.exec_())
