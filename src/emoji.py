# import tkinter as tk
# from tkinter import *
# import cv2
# from PIL import Image, ImageTk
# import os
# import numpy as np
# import cv2
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten
# from keras.layers import Conv2D
# from keras.optimizers import Adam
# from keras.layers import MaxPooling2D
# from keras.preprocessing.image import ImageDataGenerator
# import threading
#
#
# emotion_model = Sequential()
# emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
# emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
# emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
# emotion_model.add(Dropout(0.25))
# emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
# emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
# emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
# emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
# emotion_model.add(Dropout(0.25))
# emotion_model.add(Flatten())
# emotion_model.add(Dense(1024, activation='relu'))
# emotion_model.add(Dropout(0.5))
# emotion_model.add(Dense(7, activation='softmax'))
# emotion_model.load_weights('model.h5')
# cv2.ocl.setUseOpenCL(False)
#
# emotion_dict = {0: "  Angry  ", 2: "  Disgusted  ", 3: "  Fearful  ", 4: "  Neutral  ", 5: "  Sad  ", 6: "  Surprised  "}
#
# cur_path = os.path.dirname(os.path.abspath(__file__))
#
# emoji_dist = {
#     0: cur_path + "/emojis/angry.png",
#     1: cur_path + "/emojis/disgusted.png",
#     2: cur_path + "/emojis/fearful.png",
#     3: cur_path + "/emojis/happy.png",
#     4: cur_path + "/emojis/neutral.png",
#     5: cur_path + "/emojis/sad.png",
#     6: cur_path + "/emojis/surprised.png"
# }
#
# global last_frame1
# last_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
# global cap1
# show_text = [0]
# global frame_number
#
# def show_subject():
#     cap1 = cv2.VideoCapture(0)
#     if not cap1.isOpened():
#         print("Can't open the camera")
#
#     cap1.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.05)  # Disable auto exposure
#     cap1.set(cv2.CAP_PROP_EXPOSURE, 0.1)  # Set exposure value
#
#     global frame_number
#     length = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
#     frame_number += 1
#     if frame_number >= length:
#         exit
#     cap1.set(1, frame_number)
#     flag1, frame1 = cap1.read()
#     frame1 = cv2.resize(frame1, (600, 500))
#     bounding_box = cv2.CascadeClassifier('/Users/mohammedthansheer/Downloads/haarcascade_frontalface_default.xml')
#     gray_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
#     num_faces = bounding_box.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
#     for (x, y, w, h) in num_faces:
#         cv2.rectangle(frame1, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
#         roi_gray_frame = gray_frame[y:y + h, x:x + w]
#         cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
#         prediction = emotion_model.predict(cropped_img)
#         maxindex = int(np.argmax(prediction))
#         cv2.putText(frame1, emotion_dict[maxindex], (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,cv2.LINE_AA)
#         show_text[0] = maxindex
#     if flag1 is None:
#         print("Major error!")
#     elif flag1:
#         global last_frame1
#         last_frame1 = frame1.copy()
#         pic = cv2.cvtColor(last_frame1, cv2.COLOR_BGR2RGB)
#         img = Image.fromarray(pic)
#         imgtk = ImageTk.PhotoImage(image=img)
#         Imain.imgtk = imgtk
#         Imain.configure(image=imgtk)
#         root.update()
#         Imain.after(10, show_subject)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             exit()
#
# def show_avatar():
#     emoji_path = emoji_dist.get(show_text[0], None)
#     if emoji_path is None:
#         print(f"Error: No emoji path found for index {show_text[0]}")
#         return
#
#     print(f"Reading emoji image from: {emoji_path}")
#
#     frame2 = cv2.imread(emoji_path)
#     if frame2 is None:
#         print(f"Error: Unable to read emoji image from {emoji_path}")
#         return
#
#     pic2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
#     img2 = Image.fromarray(pic2)
#     imgtk2 = ImageTk.PhotoImage(image=img2)
#     Imain2.imgtk = imgtk2
#     Imain3.configure(text=emotion_dict[show_text[0]], font=('arial', 45, 'bold'))
#     Imain2.configure(image=imgtk2)
#     root.update()
#     Imain2.after(10, show_avatar)
#
#
#
# # if __name__ == '__main__':
# #     frame_number = 0
# #     root = tk.Tk()
# #     Imain = tk.Label(master=root, padx=50, bd=10)
# #     Imain2 = tk.Label(master=root, bd=10)
# #     Imain3 = tk.Label(master=root, bd=10, fg="#CDCDCD", bg='black')
# #     Imain.pack(side=LEFT)
# #     Imain.place(x=50, y=250)
# #     Imain3.pack()
# #     Imain3.place(x=960, y=250)
# #     Imain2.pack(side=RIGHT)
# #     Imain2.place(x=900, y=350)
# #
# #     root.title("Photo To Emoji")
# #     root.geometry("1400x900+100+10")
# #     root['bg'] = 'black'
# #     exitButton = Button(root, text='Quit', fg="red", command=root.destroy, font=('arial', 25, 'bold')).pack(side=BOTTOM)
# #
# #     threading.Thread(target=show_subject).start()
# #     threading.Thread(target=show_avatar).start()
# #
# #     # Move the scheduling line here
# #     Imain.after(10, show_subject)
# #
# #     root.mainloop()
# if __name__ == '__main__':
#     frame_number = 0
#     root = tk.Tk()
#     root.title("Photo To Emoji")
#     root.geometry("1400x900+100+10")
#     root['bg'] = 'black'
#
#     Imain = tk.Label(master=root, padx=50, bd=10)
#     Imain2 = tk.Label(master=root, bd=10)
#     Imain3 = tk.Label(master=root, bd=10, fg="#CDCDCD", bg='black')
#     Imain.pack(side=LEFT)
#     Imain.place(x=50, y=250)
#     Imain3.pack()
#     Imain3.place(x=960, y=250)
#     Imain2.pack(side=RIGHT)
#     Imain2.place(x=900, y=350)
#
#     exitButton = Button(root, text='Quit', fg="red", command=root.destroy, font=('arial', 25, 'bold')).pack(side=BOTTOM)
#
#     threading.Thread(target=show_subject).start()
#     threading.Thread(target=show_avatar).start()
#
#     # Move the scheduling line here
#     Imain.after(10, show_subject)
#
#     root.mainloop()


import tkinter as tk
from tkinter import *
import cv2
from PIL import Image, ImageTk
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import threading


emotion_model = Sequential()
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))
emotion_model.load_weights('model.h5')
cv2.ocl.setUseOpenCL(False)

emotion_dict = {0: "  happy  ", 2: "  happy  ", 3: "  happy  ", 4: "  anger  ", 5: "  sad  ", 6: "  sad  ", 7: "surprised "}

cur_path = os.path.dirname(os.path.abspath(__file__))

emoji_dist = {
    0: cur_path + "/emojis/angry.png",
    1: cur_path + "/emojis/disgusted.png",
    2: cur_path + "/emojis/fearful.png",
    3: cur_path + "/emojis/happy.png",
    4: cur_path + "/emojis/neutral.png",
    5: cur_path + "/emojis/sad.png",
    6: cur_path + "/emojis/surprised.png"
}

global last_frame1
last_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
global cap1
show_text = [0]
global frame_number


def show_subject():
    global frame_number
    cap1.set(1, frame_number)
    flag1, frame1 = cap1.read()
    frame1 = cv2.resize(frame1, (600, 500))
    bounding_box = cv2.CascadeClassifier('/Users/mohammedthansheer/Downloads/haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    num_faces = bounding_box.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame1, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
        prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        cv2.putText(frame1, emotion_dict[maxindex], (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)
        show_text[0] = maxindex

    if flag1 is None:
        print("Major error!")
    elif flag1:
        global last_frame1
        last_frame1 = frame1.copy()
        pic = cv2.cvtColor(last_frame1, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(pic)
        imgtk = ImageTk.PhotoImage(image=img)
        Imain.imgtk = imgtk
        Imain.configure(image=imgtk)

    frame_number += 1
    root.after(10, show_subject)  # Schedule the next update

def show_avatar():
    emoji_path = emoji_dist.get(show_text[0], None)
    if emoji_path is None:
        print(f"Error: No emoji path found for index {show_text[0]}")
        return

    print(f"Reading emoji image from: {emoji_path}")

    frame2 = cv2.imread(emoji_path)
    if frame2 is None:
        print(f"Error: Unable to read emoji image from {emoji_path}")
        return

    pic2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    img2 = Image.fromarray(pic2)
    imgtk2 = ImageTk.PhotoImage(image=img2)
    Imain2.imgtk = imgtk2
    Imain3.configure(text=emotion_dict[show_text[0]], font=('arial', 45, 'bold'))
    Imain2.configure(image=imgtk2)

    root.after(10, show_avatar)  # Schedule the next update

if __name__ == '__main__':
    frame_number = 0
    root = tk.Tk()
    root.title("Photo To Emoji")
    root.geometry("1400x900+100+10")
    root['bg'] = 'black'

    Imain = tk.Label(master=root, padx=50, bd=10)
    Imain2 = tk.Label(master=root, bd=10)
    Imain3 = tk.Label(master=root, bd=10, fg="#CDCDCD", bg='black')
    Imain.pack(side=LEFT)
    Imain.place(x=50, y=250)
    Imain3.pack()
    Imain3.place(x=960, y=250)
    Imain2.pack(side=RIGHT)
    Imain2.place(x=900, y=350)

    exitButton = Button(root, text='Quit', fg="red", command=root.destroy, font=('arial', 25, 'bold')).pack(side=BOTTOM)

    cap1 = cv2.VideoCapture(0)
    if not cap1.isOpened():
        print("Can't open the camera")

    cap1.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.05)  # Disable auto exposure
    cap1.set(cv2.CAP_PROP_EXPOSURE, 0.1)  # Set exposure value

    threading.Thread(target=show_subject).start()
    threading.Thread(target=show_avatar).start()

    root.mainloop()