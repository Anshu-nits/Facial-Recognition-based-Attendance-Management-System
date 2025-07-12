import csv
import os
import cv2
import pandas as pd

def TakeImage(l1, l2, haarcasecade_path, trainimage_path, message, err_screen, text_to_speech):
    if l1 == "" and l2 == "":
        text_to_speech("Please enter your Enrollment Number and Name.")
        return
    elif l1 == "":
        text_to_speech("Please enter your Enrollment Number.")
        return
    elif l2 == "":
        text_to_speech("Please enter your Name.")
        return

    try:
        Enrollment = l1
        Name = l2
        sampleNum = 0
        directory = f"{Enrollment}_{Name}"
        path = os.path.join(trainimage_path, directory)

        if os.path.exists(path):
            text_to_speech("Student data already exists.")
            message.configure(text="Student data already exists.")
            return

        os.makedirs(path)

        cam = cv2.VideoCapture(0)
        detector = cv2.CascadeClassifier(haarcasecade_path)

        while True:
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                sampleNum += 1
                img_name = os.path.join(path, f"{Name}_{Enrollment}_{sampleNum}.jpg")
                cv2.imwrite(img_name, gray[y:y+h, x:x+w])
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.imshow("Capturing Faces", img)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            elif sampleNum >= 50:
                break

        cam.release()
        cv2.destroyAllWindows()

        # Save details to CSV
        row = [Enrollment, Name]
        with open("StudentDetails/studentdetails.csv", "a+", newline="") as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)

        msg = f"Images Saved for Enrollment: {Enrollment}, Name: {Name}"
        message.configure(text=msg)
        text_to_speech(msg)

    except Exception as e:
        print("Error:", e)
        text_to_speech("An error occurred while capturing images.")
