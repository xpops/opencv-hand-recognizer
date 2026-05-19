import cv2
import numpy as np

def main():
    # Open Webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open webcam")
        return

    while True:
        # Read a frame
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame. Exiting ...")
            break

        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)

        # Define Region of Interest (ROI)
        roi_x, roi_y, roi_w, roi_h = 1200, 600, 400, 400
        roi = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)

        # Convert the ROI to HSV color space
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Define skin color range in HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)

        # Create a binary mask
        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        # Clean up the mask with morphological transformations
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.GaussianBlur(mask, (5, 5), 100)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Find the largest contour (assuming it's the hand)
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            # Draw the contour
            cv2.drawContours(roi, [max_contour], 0, (0, 255, 255), 2)

        # Display the frame and mask
        cv2.imshow('Hand Gesture Recognition', frame)
        cv2.imshow('Mask', mask)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

