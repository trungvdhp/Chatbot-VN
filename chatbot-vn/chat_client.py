#!/usr/bin/env python3
"""Script for pyqt chat client."""
from socket import AF_INET, socket, SOCK_STREAM
from threading import Thread
import tkinter

def receive():
    """Handles receiving of messages."""
    while True:
        try:
            msg = client_socket.recv(BUFSIZ).decode("utf8")
            msg_list.insert(tkinter.END, msg)
        except OSError:  # Possibly client has left the chat.
            break


def send(event=None):  # event is passed by binders.
    """Handles sending of messages."""
    msg = my_msg.get()
    my_msg.set("")  # Clears input field.
    client_socket.send(bytes(msg, "utf8"))
    if msg == "{quit}":
        client_socket.close()
        chatter.quit()


def on_closing(event=None):
    """This function is to be called when the window is closed."""
    my_msg.set("{quit}")
    send()

def text_hide(click):
    if entry_field.get() == "Type your message here":
        entry_field.delete(0, "end")      
        entry_field.insert(0, '')
        
chatter = chatci.ChatCI()
chatter.resizable(0, 0)
chatter.geometry("600x400")
chatter.title("Live chat with BOT")
chatter.configure(bg="light blue")

messages_frame = tkinter.Frame(chatter)
my_msg = tkinter.StringVar()  # For the messages to be sent.
my_msg.set("Type your message here")
scrollbar = tkinter.Scrollbar(messages_frame)  # To navigate through past messages.
# Following will contain the messages.
msg_list = tkinter.Listbox(messages_frame, height=20, width=100, yscrollcommand=scrollbar.set)
scrollbar.pack(side=tkinter.RIGHT, fill=tkinter.Y)
msg_list.pack(side=tkinter.LEFT, fill=tkinter.BOTH)
msg_list.pack()
messages_frame.pack()

entry_field = tkinter.Entry(chatter, width=100, font=('Arial',11), textvariable=my_msg)
entry_field.bind('<FocusIn>', text_hide)
entry_field.bind("<Return>", send)
entry_field.pack()
send_button = tkinter.Button(chatter, text="Send", command=send)
send_button.pack()

chatter.protocol("WM_DELETE_WINDOW", on_closing)

#----Now comes the sockets part----
HOST = input('Enter host: ')
PORT = input('Enter port: ')
if not PORT:
    PORT = 33000
else:
    PORT = int(PORT)

BUFSIZ = 1024
ADDR = (HOST, PORT)

client_socket = socket(AF_INET, SOCK_STREAM)
client_socket.connect(ADDR)

receive_thread = Thread(target=receive)
receive_thread.start()
tkinter.mainloop()  # Starts GUI execution.