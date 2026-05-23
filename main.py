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

        # Get frame dimensions to ensure safe ROI bounds
        fh, fw, _ = frame.shape

        # Define Region of Interest (ROI)
        roi_w, roi_h = 400, 400
        # Dynamically place ROI on the right side of the screen safely within bounds
        roi_x = max(0, min(1200, fw - roi_w - 10))
        roi_y = max(0, min(600, fh - roi_h - 10))
        roi = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)

        # Pre-process the ROI with a Gaussian blur to reduce high-frequency salt-and-pepper noise
        roi_blurred = cv2.GaussianBlur(roi, (5, 5), 0)

        # Convert the blurred ROI to HSV and YCrCb color spaces
        hsv = cv2.cvtColor(roi_blurred, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(roi_blurred, cv2.COLOR_BGR2YCrCb)

        # Define skin color range in HSV
        lower_hsv = np.array([0, 20, 70], dtype=np.uint8)
        upper_hsv = np.array([20, 255, 255], dtype=np.uint8)

        # Define skin color range in YCrCb (extremely robust against gray/greenish background walls)
        lower_ycrcb = np.array([0, 133, 77], dtype=np.uint8)
        upper_ycrcb = np.array([255, 173, 127], dtype=np.uint8)

        # Create binary masks for both color spaces
        mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)
        mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)

        # Combine the masks using bitwise AND (a pixel must be classified as skin in both spaces)
        mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)

        # Clean up the mask using morphological operations
        # Using an elliptical structuring element which matches hand shapes better than a square
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # Morphological opening to remove small background noise speckles
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        # Morphological closing to fill small holes/gaps inside the detected hand area
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

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

