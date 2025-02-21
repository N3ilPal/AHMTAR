import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe Pose and Hands.
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
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

# Initialize the pose and hands modules with higher confidence thresholds.
def initialize_modules():
    pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    return pose, hands

# Process a single frame for pose and hands.
def process_frame(pose, hands, frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(frame_rgb)
    hands_results = hands.process(frame_rgb)
    return pose_results, hands_results

# Display a message in the frame.
def display_message(image, message_lines, y_offset=50, color=(255, 0, 0)):
    for line in message_lines:
        cv2.putText(image, line, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        y_offset += 30
    return image

# Process tests and determine if the test is completed.
def process_tests(pose_results, hands_results, current_test, fist_open_close_count, fist_threshold):
    test_completed = False
    if current_test == 0:  # Raise both arms above head
        if pose_results.pose_landmarks:
            left_wrist = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
            left_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            if left_wrist.y < left_shoulder.y and right_wrist.y < right_shoulder.y:
                test_completed = True

    elif current_test in [1, 2]:  # Close and open right/left fist 5 times
        if hands_results.multi_hand_landmarks:
            hand_landmarks = hands_results.multi_hand_landmarks[0].landmark
            thumb_tip = hand_landmarks[mp_hands.HandLandmark.THUMB_TIP]
            index_finger_tip = hand_landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            fist_distance = calculate_distance(thumb_tip, index_finger_tip)
            if fist_distance < fist_threshold:
                if fist_open_close_count % 2 == 0:
                    fist_open_close_count += 1
            elif fist_distance > fist_threshold:
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

    return test_completed, fist_open_close_count

# Main loop to run the tests.
def run_tests():
    pose, hands = initialize_modules()
    cap = cv2.VideoCapture(0)
    task_start_time = time.time()
    buffer_end_time = 0
    current_test = -1
    fist_open_close_count = 0
    fist_threshold = 0.1  # Threshold for fist open/close detection
    test_results = []
    welcome_message = "Welcome! We are now going to begin our Advanced Hand Movement Tracking and Analysis. Please follow the instructions for accurate results."
    welcome_lines = wrap_text(welcome_message, 7)
    post_arms_buffer = False  # New flag to check if the "Put your arms down" message is shown

    tests = [
        "Raise both arms above your head",
        "Close and open your right fist 5 times",
        "Close and open your left fist 5 times",
        "Move your right hand from right to left",
        "Move your left hand from left to right"
    ]

    try:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            image = cv2.flip(image, 1)
            pose_results, hands_results = process_frame(pose, hands, image)

            # Display welcome message for the first 5 seconds.
            if current_test == -1:
                if time.time() - task_start_time < 5:
                    image = display_message(image, welcome_lines)
                else:
                    current_test = 0  # Start the first test after buffer period
                    task_start_time = time.time()
                    buffer_end_time = task_start_time + 15

            elif current_test < len(tests):
                if time.time() < buffer_end_time:
                    buffer_message = "Get ready for the next test..."
                    image = display_message(image, [buffer_message])
                else:
                    if post_arms_buffer:  # Show "Put your arms down" message
                        if time.time() - task_start_time < 5:  # Show for 5 seconds
                            image = display_message(image, ["Put your arms down"])
                        else:
                            post_arms_buffer = False
                            current_test += 1
                            task_start_time = time.time()
                            buffer_end_time = task_start_time + 15
                            fist_open_close_count = 0  # Reset for next test

                    else:
                        test_completed, fist_open_close_count = process_tests(
                            pose_results, hands_results, current_test, fist_open_close_count, fist_threshold)

                        # If test 0 (raise arms) is completed, show buffer message to put arms down
                        if current_test == 0 and test_completed:
                            post_arms_buffer = True
                            test_completed = False
                            task_start_time = time.time()

                        # Move to the next test if completed or time has expired
                        if test_completed or (time.time() - task_start_time) > 30:
                            test_results.append(test_completed)
                            current_test += 1
                            task_start_time = time.time()
                            buffer_end_time = task_start_time + 15
                            fist_open_close_count = 0  # Reset for next test

                        if current_test < len(tests):
                            image = display_message(image, wrap_text(tests[current_test], 7))

                            if current_test in [1, 2]:  # Show fist open/close count
                                image = display_message(image, [f'Fist Open/Close Count: {fist_open_close_count // 2}'], y_offset=200)

            else:
                # Display the final result message
                final_result = all(test_results)
                result_text = "You passed all tests. No rehabilitation needed." if final_result else "Further rehabilitation may be required."
                image = display_message(image, wrap_text(result_text, 7), y_offset=100)
                cv2.imshow('Mobility Tests', image)
                cv2.waitKey(30000)
                break

            cv2.imshow('Mobility Tests', image)
            if cv2.waitKey(5) & 0xFF == 27:  # Exit on pressing 'ESC'
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

# Run the tests
run_tests()
