# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'nemom.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1039, 633)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_4.setObjectName("gridLayout_4")
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
        self.tab_temp = QtWidgets.QWidget()
        self.tab_temp.setObjectName("tab_temp")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.tab_temp)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.temp_inner = QtWidgets.QTabWidget(self.tab_temp)
        self.temp_inner.setObjectName("temp_inner")
        self.gridLayout_2.addWidget(self.temp_inner, 0, 0, 1, 1)
        self.tabWidget.addTab(self.tab_temp, "")
        self.tab_edges_old = QtWidgets.QWidget()
        self.tab_edges_old.setObjectName("tab_edges_old")
        self.gridLayout = QtWidgets.QGridLayout(self.tab_edges_old)
        self.gridLayout.setObjectName("gridLayout")
        self.edges_inner_old = QtWidgets.QTabWidget(self.tab_edges_old)
        self.edges_inner_old.setObjectName("edges_inner_old")
        self.gridLayout.addWidget(self.edges_inner_old, 0, 0, 1, 1)
        self.tabWidget.addTab(self.tab_edges_old, "")
        self.gridLayout_4.addWidget(self.tabWidget, 0, 0, 1, 1)
        self.tabWidget_2 = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget_2.setObjectName("tabWidget_2")
        self.tab_edges = QtWidgets.QWidget()
        self.tab_edges.setObjectName("tab_edges")
        self.gridLayout1 = QtWidgets.QGridLayout(self.tab_edges)
        self.gridLayout1.setObjectName("gridLayout1")
        self.edges_inner = QtWidgets.QTabWidget(self.tab_edges)
        self.edges_inner.setObjectName("edges_inner")
        self.gridLayout1.addWidget(self.edges_inner, 0, 0, 1, 1)
        self.tabWidget_2.addTab(self.tab_edges, "")
        self.tab_drawing = QtWidgets.QWidget()
        self.tab_drawing.setObjectName("tab_drawing")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.tab_drawing)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.drawing = QtWidgets.QLabel(self.tab_drawing)
        self.drawing.setText("")
        self.drawing.setObjectName("drawing")
        self.horizontalLayout.addWidget(self.drawing)
        self.tabWidget_2.addTab(self.tab_drawing, "")
        self.gridLayout_4.addWidget(self.tabWidget_2, 0, 1, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(1)
        self.tabWidget_2.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Nao-EMOM"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_camera), _translate("MainWindow", "Camera"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_temp), _translate("MainWindow", "Temp"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_edges_old), _translate("MainWindow", "Edges"))
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self.tab_edges), _translate("MainWindow", "Edges"))
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self.tab_drawing), _translate("MainWindow", "Geometry"))

