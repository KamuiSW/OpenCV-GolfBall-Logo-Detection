import cv2
import timeit
import numpy as np
import features
import keyboard
import RPI.GPIO as GPIO
from time import sleep
GPIO.setmode(GPIO.BOARD)

#Motor vars
motorTime = 10
waitingSecondMotor = 10
waitingSecondMotorY = 10

#Motor GPIO setup
GPIO.setup(11,GPIO.out)
motx = GPIO.PWM(11, 50)
motx.start(0)

# LOGOOOOOOOOOOOOS
logos = {
    "logo1": {
        "image": "Logo/pinnacle.png",
        "action": lambda: print("Detected Pinnacle golf ball!"),
    },
    "logo2": {
        "image": "Logo/wil.png",   
        "action": lambda: print("Detected logo Wilson!"),
    },
    "logo3": {
        "image": "Logo/pinnacle.jpg",
        "action": lambda: print("Detected Pinnacle golf ball!(stom versie)"),
    },
    "logo4": {
        "image": "Logo/cal1.png",
        "action": lambda: print("Detected cal!"),
    },
    "logo5": {
        "image": "Logo/Tileist.png",
        "action": lambda: print("Detected Tit!"),
    },
}

def load_logo_features():
    # Load features
    for key, data in logos.items():
        img = cv2.imread(data["image"])
        logos[key]["features"] = features.getFeatures(img)

def motorRotationX():
    motx.ChangeDutyCycle(3)
    
def motorRotationY():
    motx.ChangeDutyCycle(3)

while True: #Motor phase one to second handler
    waitingSecondMotor-=1
    if(waitingSecondMotor >= 0):
        motorRotationX()
    if(waitingSecondMotor == 0):
        waitingSecondMotorY -=1
        

    break

def main():
    
    load_logo_features()  # Load features

    video_src = 0
    cam = cv2.VideoCapture(video_src)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 500)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)
    
    
    cur_time = timeit.default_timer()
    frame_number = 0
    scan_fps = 0

    while True:
        frame_got, frame = cam.read()
        if not frame_got:
            break

        frame_number += 1
        if not frame_number % 100:
            scan_fps = 1 / ((timeit.default_timer() - cur_time) / 100)
            cur_time = timeit.default_timer()

        # Try to detect them logo's in each f
        detected_logo = None
        for logo_name, data in logos.items():
            region = features.detectFeatures(frame, data["features"])
            if region is not None:
                detected_logo = logo_name
                # cool box around detected logo
                box = cv2.boxPoints(region)
                box = np.int0(box)
                cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)
                # execute stuff that i want for the logo
                data["action"]()
                break

        # Show FPS on the frame
        cv2.putText(frame, f'FPS {scan_fps:.3f}', org=(0, 50),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    fontScale=1, color=(0, 0, 255))
        
        # Show the video preview
        cv2.imshow("mamba out", frame)
        if cv2.waitKey(10) == 27:
            break

if __name__ == '__main__':
    main()
