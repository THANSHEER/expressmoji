import tkinter as tk
from tkinter import *
import cv2
from PIL import Image, ImageTk
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import threading
import time

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

emotion_dict = {0: "happy", 2: "angry", 3: "happy", 4: "anger", 5: "sad", 6: "sad", 7: "surprised"}

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

# Variables for accumulating predictions
accumulated_predictions = []

# Variable to control the recognizer freeze
recognizer_frozen = False


def check_expression(countdown_label):
    global recognizer_frozen

    # Unfreeze recognizer
    recognizer_frozen = False

    # Start the countdown timer
    for i in range(15, 0, -1):
        countdown_label.config(text=f"Recognizing in {i} seconds")
        root.update()
        time.sleep(1)

    # Reset the countdown label after 15 seconds
    countdown_label.config(text="")

    # Make a decision every 15 seconds
    if accumulated_predictions:
        final_prediction = max(set(accumulated_predictions), key=accumulated_predictions.count)
        print(f"Decision after 15 seconds: {emotion_dict[final_prediction]}")
        accumulated_predictions.clear()

    # Schedule the next check after 15 seconds
    root.after(0, check_expression, countdown_label)


def show_subject():
    global frame_number, recognizer_frozen
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

        if not recognizer_frozen:
            prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            cv2.putText(frame1, emotion_dict[maxindex], (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                        2,
                        cv2.LINE_AA)
            show_text[0] = maxindex

            # Accumulate predictions
            accumulated_predictions.append(maxindex)

    if frame_number % 10 == 0 and accumulated_predictions:
        # Make a decision after 10 frames
        final_prediction = max(set(accumulated_predictions), key=accumulated_predictions.count)
        print(f"Decision after 10 frames: {emotion_dict[final_prediction]}")
        accumulated_predictions.clear()

        # Freeze recognizer for the next 15 seconds
        recognizer_frozen = True

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

    countdown_label = tk.Label(root, text="", font=('arial', 20, 'bold'), bg='black', fg='white')
    countdown_label.pack(side=BOTTOM)

    threading.Thread(target=show_subject).start()
    threading.Thread(target=show_avatar).start()

    # Schedule the first check after 15 seconds
    root.after(15000, check_expression, countdown_label)

    root.mainloop()
