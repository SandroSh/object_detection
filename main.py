import cv2
from Canny import  canny

cap = cv2.VideoCapture("bData2.mp4")
previous_frame = None


cv2.namedWindow('Mask', cv2.WINDOW_NORMAL)
cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Frame', 600, 400)
cv2.resizeWindow('Mask', 700, 400)

tracked_objects = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if previous_frame is not None:
        diff = cv2.absdiff(gray_frame, previous_frame)
        # edges = canny(diff, 50, 150)
        edges = cv2.Canny(diff, 30, 100)

        # Generating Matrix
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        # increasing the size of bright regions
        edges = cv2.dilate(edges, kernel, iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        bounding_boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 0:
                x, y, w, h = cv2.boundingRect(cnt)
                bounding_boxes.append([x, y, w, h])

        for (x, y, w, h) in bounding_boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('Mask', edges)

    previous_frame = gray_frame.copy()

    cv2.imshow('Frame', frame)

    if cv2.waitKey(10) == 27:
        break

cap.release()
cv2.destroyAllWindows()


