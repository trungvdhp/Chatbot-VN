# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 09:25:47 2018

@author: trung
"""

import sys
from socket import AF_INET, socket, SOCK_STREAM
from threading import Thread
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QPushButton, QAction, \
QLineEdit, QMessageBox, QListWidget, QGridLayout, QHBoxLayout, QVBoxLayout, QFormLayout, \
QComboBox, QLabel
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot, pyqtSignal, Qt
 
class ChatCI(QMainWindow):
    # define new signal
    resized = pyqtSignal()
    def __init__(self):
        super().__init__()
        self.title = 'Live chat with BOT'
        self.left = 0
        self.top = 0
        self.margin = 10
        self.width = 400
        self.height = 500
        self.msg = ""
        #----Now comes the sockets part----
        self.HOST = input('Enter host: ')
        self.PORT = input('Enter port: ')
        if not self.PORT:
            self.PORT = 33000
        else:
            self.PORT = int(self.PORT)

        self.BUFSIZ = 1024
        self.ADDR = (self.HOST, self.PORT)
    
        self.client_socket = socket(AF_INET, SOCK_STREAM)
        self.client_socket.connect(self.ADDR)
        self.initUI()
        self.receive_thread = Thread(target=self.receive)
        self.receive_thread.start()
        
    # init user interface of QWidget
    def initUI(self):
        self.setWindowTitle(self.title)
        #self.setFixedSize(self.width, self.height)
        self.setMinimumHeight(self.height)
        self.setMinimumWidth(self.width)
        self.setStyleSheet("QMainWindow {background: 'light blue';}");
        self.resized.connect(self.on_resize)
        
        # Create listbox
        self.listWidget = QListWidget(self)
        self.listWidget.move(self.margin, self.margin)
        self.listWidget.model().rowsInserted.connect(self.on_rowsInsert)
        # Add some sample items
        for i in range(1,1):
            self.listWidget.addItem("Item #" + str(i))
        self.listWidget.show()
        self.listWidget.scrollToBottom()
        
        # Create a button in the window
        self.button = QPushButton('Send', self)
        self.button.setAutoDefault(True)
        # connect button to function on_click
        self.button.clicked.connect(self.on_click) 
        
        # Create textbox above button
        self.textbox = QLineEdit(self)
        self.textbox.setPlaceholderText("Type your message here")
        self.textbox.setClearButtonEnabled(True)
        self.textbox.setTextMargins(2, 2, 2, 2)
        # connect texbox to function on_click of button
        self.textbox.returnPressed.connect(self.button.click)
        # set focus in textbox
        self.textbox.setFocus()
        
        self.show()
 
    # 
    def on_resize(self):
        width = self.frameSize().width()
        height = self.frameSize().height()
        self.listWidget.resize(width - 3.5*self.margin, height - 11.5*self.margin)
        self.textbox.move(self.margin, height - 10*self.margin)
        self.textbox.resize(width-3.5*self.margin, 2.5*self.margin)
        self.button.move(width/2 - 3*self.margin, height - 7*self.margin)
        self.button.resize(6*self.margin, 2.5*self.margin)
        self.listWidget.scrollToBottom()
    
    # receive message from server
    def receive(self):
        """Handles receiving of messages."""
        while True:
            try:
                self.msg = self.client_socket.recv(self.BUFSIZ).decode("utf8")
                self.listWidget.addItem(self.msg)
            except OSError:  # Possibly client has left the chat.
                break

    # on clicked, add item to listbox; clear and focus textbox
    def on_click(self):
        """Handles sending of messages."""
        self.msg = self.textbox.text()
        # Clear input field
        self.textbox.setText("")
        self.textbox.setFocus()
        
        if self.msg == "{quit}":
            self.close()
        else:
            try:
                self.client_socket.send(bytes(self.msg, "utf8"))
            except OSError:
                 QMessageBox.warning(self, 
                                     "Error on sending message to server", 
                                     "Server connection is interupted")
                 
    # on rows inserted, scroll to bottom of list
    def on_rowsInsert(self):
        self.listWidget.scrollToBottom()
    
    # overidde resizeEvent of QMainWindow
    def resizeEvent(self, event):
        self.resized.emit()
        return super(ChatCI, self).resizeEvent(event)
    
    # Before close qmainwindow ask to confirm and if ok then send {quit} to server
    def closeEvent(self,event):
        print("Closing")
        result = QMessageBox.question(self,
                      "Confirm Exit...",
                      "Are you sure you want to exit ?",
                      QMessageBox.Yes| QMessageBox.No)
        event.ignore()

        if result == QMessageBox.Yes: 
            try:
                self.client_socket.send(bytes(self.msg, "utf8"))
            except OSError:
                 QMessageBox.warning(self, 
                                     "Error on sending message to server", 
                                     "Server connection is interupted")
            self.client_socket.close()
            event.accept()
            print("Closed")
        
# main app
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ChatCI()
    sys.exit(app.exec_())