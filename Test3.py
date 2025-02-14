import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe Pose and Hands.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# For drawing landmarks on the image.
mp_drawing = mp.solutions.drawing_utils

# Define a function to calculate the distance between two points.
def calculate_distance(point1, point2):
    return np.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2 + (point1.z - point2.z) ** 2)

# Define a function to calculate the horizontal distance between two points.
def calculate_horizontal_distance(point1, point2):
    return abs(point1.x - point2.x)

# Function to wrap text into lines with a maximum of `max_words_per_line` words per line.
def wrap_text(text, max_words_per_line):
    words = text.split()
    lines = []
    for i in range(0, len(words), max_words_per_line):
        lines.append(" ".join(words[i:i + max_words_per_line]))
    return lines

# Setup video capture.
cap = cv2.VideoCapture(0)

# Test parameters
tests = [
    "Raise both arms above your head",
    "Close and open your right fist 5 times",
    "Close and open your left fist 5 times",
    "Move your right hand from right to left",
    "Move your left hand from left to right"
]
current_test = -1
test_completed = False
fist_open_close_count = 0
fist_threshold = 0.1  # Distance change threshold to detect fist open/close.
task_start_time = time.time()
buffer_end_time = 0  # Time when the current buffer period ends
test_results = []

welcome_message = "Welcome, we are now going to begin our Advanced Hand Movement Tracking and Analysis for Rehabilitation Testing. Please follow the instructions on the screen to provide the most accurate results."
welcome_lines = wrap_text(welcome_message, 7)

print(welcome_message)

try:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Invert the image (flip horizontally).
        image = cv2.flip(image, 1)

        # Convert the BGR image to RGB, and process it with MediaPipe Pose and Hands.
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(image_rgb)
        hands_results = hands.process(image_rgb)

        # Convert the image back to BGR for OpenCV.
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # Display the welcome message for the first 5 seconds.
        if current_test == -1:
            if time.time() - task_start_time < 5:
                y_offset = 50
                for line in welcome_lines:
                    cv2.putText(image, line, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
                    y_offset += 30
            else:
                current_test = 0  # Set to 0 to start the 15-second buffer before the first test
                task_start_time = time.time()
                buffer_end_time = task_start_time + 15

        elif current_test < len(tests):
            # Check if the buffer time has passed before starting the test.
            if time.time() < buffer_end_time:
                buffer_message = "Get ready for the next test..."
                cv2.putText(image, buffer_message, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
            else:
                # Check if the current test is completed or if time is up.
                if test_completed or (time.time() - task_start_time) > 30:
                    if current_test >= 0:
                        test_results.append(test_completed)
                    current_test += 1
                    test_completed = False
                    fist_open_close_count = 0  # Reset fist count for next test
                    task_start_time = time.time()
                    buffer_end_time = task_start_time + 15

                if current_test < len(tests):
                    # Draw pose landmarks on the image.
                    if pose_results.pose_landmarks:
                        mp_drawing.draw_landmarks(image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                    # Draw hand landmarks on the image.
                    if hands_results.multi_hand_landmarks:
                        for hand_landmarks in hands_results.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Test logic
                    if current_test == 0:  # Raise both arms above head
                        if pose_results.pose_landmarks:
                            left_wrist = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
                            right_wrist = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
                            left_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
                            right_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

                            if left_wrist.y < left_shoulder.y and right_wrist.y < right_shoulder.y:
                                test_completed = True

                    elif current_test == 1:  # Close and open right fist 5 times
                        if hands_results.multi_hand_landmarks:
                            right_fist_distance = calculate_distance(
                                hands_results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.THUMB_TIP],
                                hands_results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],
                            )

                            if right_fist_distance < fist_threshold:
                                if fist_open_close_count % 2 == 0:
                                    fist_open_close_count += 1
                            elif right_fist_distance > fist_threshold:
                                if fist_open_close_count % 2 == 1:
                                    fist_open_close_count += 1
                                    if fist_open_close_count // 2 >= 5:
                                        test_completed = True

                    elif current_test == 2:  # Close and open left fist 5 times
                        if hands_results.multi_hand_landmarks:
                            left_fist_distance = calculate_distance(
                                hands_results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.THUMB_TIP],
                                hands_results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],
                            )

                            if left_fist_distance < fist_threshold:
                                if fist_open_close_count % 2 == 0:
                                    fist_open_close_count += 1
                            elif left_fist_distance > fist_threshold:
                                if fist_open_close_count % 2 == 1:
                                    fist_open_close_count += 1
                                    if fist_open_close_count // 2 >= 5:
                                        test_completed = True

                    elif current_test == 3:  # Move right hand from right to left
                        if pose_results.pose_landmarks:
                            right_wrist = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
                            nose = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]

                            if right_wrist.x < nose.x:
                                test_completed = True

                    elif current_test == 4:  # Move left hand from left to right
                        if pose_results.pose_landmarks:
                            left_wrist = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
                            nose = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]

                            if left_wrist.x > nose.x:
                                test_completed = True

                    # Display instructions and counters.
                    instructions_lines = wrap_text(tests[current_test], 7)
                    y_offset = 50
                    for line in instructions_lines:
                        cv2.putText(image, line, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                        y_offset += 40

                    if current_test in [1, 2]:  # Show fist open/close count
                        cv2.putText(image, f'Fist Open/Close Count: {fist_open_close_count // 2}', (20, y_offset + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        else:
            # Display the results
            if all(test_results):
                result_text = "You were able to pass. No rehabilitation is seemed to be needed at this point."
            else:
                result_text = "You seem to need extra rehabilitation. Please reach out to a medical professional."
            result_lines = wrap_text(result_text, 7)
            y_offset = 100
            for line in result_lines:
                cv2.putText(image, line, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                y_offset += 40
            cv2.imshow('Mobility Tests', image)
            cv2.waitKey(30000)  # Wait for 30 seconds before closing
            break

        # Show the image with the instructions overlay.
 #       cv2.imshow('Mobility Tests', image)
#        if cv2.waitKey(5) & 0xFF == 27:  # Exit on pressing 'ESC'
 #           break

finally:
    # Release the video capture object and close all OpenCV windows.
    cap.release()
    cv2.destroyAllWindows()
