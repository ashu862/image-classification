import streamlit as st
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

st.title("Object Measurement with Webcam")

# Global variable to track whether Streamlit app is running
streamlit_running = True

# Function to measure objects
def measure_objects():
    cap = cv2.VideoCapture(0)
    while streamlit_running:
        ref, frame = cap.read()
        frame = cv2.resize(frame, None, fx=1, fy=1, interpolation=cv2.INTER_AREA)
        orig = frame.copy()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (15, 15), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        kernel = np.ones((3, 3), np.uint8)
        closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)

        result_img = closing.copy()
        contours, hierarchy = cv2.findContours(result_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        object_count = 0
        pixelsPerMetric = None

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1000 or area > 120000:
                continue

            box = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
            box = np.array(box, dtype="int")
            box = perspective.order_points(box)
            cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 64), 2)

            for (x, y) in box:
                cv2.circle(orig, (int(x), int(y)), 5, (0, 255, 64), -1)

            (tl, tr, br, bl) = box
            (tltrX, tltrY) = midpoint(tl, tr)
            (blbrX, blbrY) = midpoint(bl, br)
            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)

            lebar_pixel = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            panjang_pixel = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

            if pixelsPerMetric is None:
                pixelsPerMetric = lebar_pixel
                pixelsPerMetric = panjang_pixel
            lebar = lebar_pixel
            panjang = panjang_pixel

            cv2.putText(orig, "L: {:.1f}CM".format(lebar_pixel / 25.5), (int(trbrX + 10), int(trbrY)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(orig, "P: {:.1f}CM".format(panjang_pixel / 25.5), (int(tltrX - 15), int(tltrY - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            object_count += 1

        cv2.putText(orig, "Distance:1-FT: {}".format(object_count), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow('Webcam', orig)

        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# Main Streamlit app
option = st.sidebar.selectbox("Select an option", ["Start Webcam", "Exit"])

if option == "Start Webcam":
    st.write("Press the button to start the webcam.")
    if st.button("Start"):
        measure_objects()

elif option == "Exit":
    streamlit_running = False  # Stop the webcam loop
