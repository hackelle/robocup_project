# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'saeros.ui'
#
# Created by: PyQt5 UI code generator 5.11.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1039, 633)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName("tabWidget")
        self.tab_camera = QtWidgets.QWidget()
        self.tab_camera.setObjectName("tab_camera")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.tab_camera)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.camera = QtWidgets.QLabel(self.tab_camera)
        self.camera.setText("")
        self.camera.setObjectName("camera")
        self.horizontalLayout_5.addWidget(self.camera)
        self.tabWidget.addTab(self.tab_camera, "")
        self.tab_edges = QtWidgets.QWidget()
        self.tab_edges.setObjectName("tab_edges")
        self.gridLayout = QtWidgets.QGridLayout(self.tab_edges)
        self.gridLayout.setObjectName("gridLayout")
        self.edges = QtWidgets.QLabel(self.tab_edges)
        self.edges.setText("")
        self.edges.setObjectName("edges")
        self.gridLayout.addWidget(self.edges, 0, 0, 1, 1)
        self.tabWidget.addTab(self.tab_edges, "")
        self.tab_temp = QtWidgets.QWidget()
        self.tab_temp.setObjectName("tab_temp")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.tab_temp)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.temp = QtWidgets.QLabel(self.tab_temp)
        self.temp.setText("")
        self.temp.setObjectName("temp")
        self.gridLayout_2.addWidget(self.temp, 0, 0, 1, 1)
        self.tabWidget.addTab(self.tab_temp, "")
        self.horizontalLayout_4.addWidget(self.tabWidget)
        self.tabWidget_2 = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget_2.setObjectName("tabWidget_2")
        self.tab_5 = QtWidgets.QWidget()
        self.tab_5.setObjectName("tab_5")
        self.tabWidget_2.addTab(self.tab_5, "")
        self.tab_6 = QtWidgets.QWidget()
        self.tab_6.setObjectName("tab_6")
        self.tabWidget_2.addTab(self.tab_6, "")
        self.horizontalLayout_4.addWidget(self.tabWidget_2)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(2)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Saeros"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_camera), _translate("MainWindow", "Camera"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_edges), _translate("MainWindow", "Edges"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_temp), _translate("MainWindow", "Temp"))
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self.tab_5), _translate("MainWindow", "Tab 1"))
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self.tab_6), _translate("MainWindow", "Tab 2"))

