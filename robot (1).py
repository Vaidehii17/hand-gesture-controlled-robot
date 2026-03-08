import serial
import cv2
import mediapipe as mp
import time
import math
from collections import deque

# ------------------- Configuration -------------------
write_video = True
debug = False
cam_source = 0  # 0,1 for USB cam, or "http://..." for IP webcam

# Try to open serial port; if it fails, set debug=True and continue
ser = None
if not debug:
    try:
        ser = serial.Serial('COM3', 115200, timeout=1)
        print(f"Serial port COM3 opened successfully")
    except serial.SerialException as e:
        print(f"Warning: Could not open COM4: {e}")
        print("Running in DEBUG mode (angles will be printed but not sent).")
        debug = True

# ---- Servo angle limits (in degrees) ----
# Base (servo 0) - Controlled by hand X position
x_min = 0
x_mid = 90
x_max = 180
hand_x_min = -0.2
hand_x_max = 0.2

# Shoulder (servo 1) - Controlled by wrist Y position
y_min = 0
y_mid = 90
y_max = 180
wrist_y_min = 0.3
wrist_y_max = 0.9

# Elbow (servo 2) - Controlled by palm size
z_min = 10
z_mid = 90
z_max = 180
palm_size_min = 0.1
palm_size_max = 0.3

# Wrist (servo 3) - Controlled by INDEX & MIDDLE FINGER EXTENSION
wrist_min = 0
wrist_mid = 60
wrist_max = 120
# Finger extension range (0 = fully curled, 1 = fully extended)
finger_extension_min = 0.3   # Fingers curled
finger_extension_max = 0.8   # Fingers extended

# Gripper (servo 4) - Controlled by thumb-index pinch
gripper_min = 0
gripper_mid = 90
gripper_max = 180
pinch_dist_min = 0.02
pinch_dist_max = 0.15

# ---- SPEED CONTROL SETTINGS ----
MAX_SPEED = 3
MIN_SPEED = 1
ACCELERATION = 0.2

# ---- Home position ----
HOME_POSITION = [x_mid, y_mid, z_mid, wrist_mid, gripper_mid]

# ---- Smoothing and dead zone ----
smoothing_alpha = 0.3
dead_zone = 1

# ---- Auto-stop settings ----
NO_HAND_TIMEOUT = 0.5
last_hand_detected_time = time.time()
hand_present = False

# ---- Movement history for speed control ----
target_angles = HOME_POSITION.copy()
current_angles = HOME_POSITION.copy()
prev_angles = HOME_POSITION.copy()
angle_velocities = [0.0] * 5

# ------------------------------------------------------

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(cam_source)

# Video writer
if write_video:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 60.0, (640, 480))

clamp = lambda n, minn, maxn: max(min(maxn, n), minn)
map_range = lambda x, in_min, in_max, out_min, out_max: int((x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)

def calculate_finger_extension(hand_landmarks, finger_tip_id, finger_mcp_id, finger_pip_id):
    """Calculate how extended a finger is (0 = curled, 1 = extended)"""
    tip = hand_landmarks.landmark[finger_tip_id]
    pip = hand_landmarks.landmark[finger_pip_id]
    mcp = hand_landmarks.landmark[finger_mcp_id]
    wrist = hand_landmarks.landmark[0]
    
    # Calculate distances
    tip_to_pip = ((tip.x - pip.x)**2 + (tip.y - pip.y)**2 + (tip.z - pip.z)**2)**0.5
    pip_to_mcp = ((pip.x - mcp.x)**2 + (pip.y - mcp.y)**2 + (pip.z - mcp.z)**2)**0.5
    mcp_to_wrist = ((mcp.x - wrist.x)**2 + (mcp.y - wrist.y)**2 + (mcp.z - wrist.z)**2)**0.5
    
    # When finger is extended, tip-pip distance is larger relative to mcp-wrist
    extension_ratio = tip_to_pip / mcp_to_wrist
    
    # Normalize to 0-1 range (empirical values)
    normalized = clamp((extension_ratio - 0.1) / 0.3, 0, 1)
    
    return normalized

def get_index_middle_extension(hand_landmarks):
    """Get combined extension of index and middle fingers"""
    # Index finger landmarks: tip=8, pip=6, mcp=5
    index_ext = calculate_finger_extension(hand_landmarks, 8, 5, 6)
    
    # Middle finger landmarks: tip=12, pip=10, mcp=9
    middle_ext = calculate_finger_extension(hand_landmarks, 12, 9, 10)
    
    # Average for combined control
    combined_ext = (index_ext + middle_ext) / 2
    
    return combined_ext, index_ext, middle_ext

def landmark_to_servo_angle(hand_landmarks):
    """Convert hand landmarks to target servo angles with two-finger wrist control"""
    wrist = hand_landmarks.landmark[0]
    index_mcp = hand_landmarks.landmark[5]
    index_tip = hand_landmarks.landmark[8]
    thumb_tip = hand_landmarks.landmark[4]
    middle_mcp = hand_landmarks.landmark[9]

    # Calculate palm center
    palm_center_x = (wrist.x + index_mcp.x + middle_mcp.x) / 3
    palm_center_y = (wrist.y + index_mcp.y + middle_mcp.y) / 3
    
    # Palm size
    palm_size = ((wrist.x - index_mcp.x)**2 +
                 (wrist.y - index_mcp.y)**2 +
                 (wrist.z - index_mcp.z)**2)**0.5

    # ---- BASE from hand X position ----
    hand_x = palm_center_x - 0.5
    hand_x = clamp(hand_x, hand_x_min, hand_x_max)
    base = map_range(hand_x, hand_x_min, hand_x_max, x_min, x_max)

    # ---- SHOULDER from wrist Y position ----
    wrist_y = clamp(wrist.y, wrist_y_min, wrist_y_max)
    shoulder = map_range(wrist_y, wrist_y_min, wrist_y_max, y_max, y_min)

    # ---- ELBOW from palm size ----
    p_size = clamp(palm_size, palm_size_min, palm_size_max)
    elbow = map_range(p_size, palm_size_min, palm_size_max, z_min, z_max)

    # ---- WRIST from INDEX & MIDDLE FINGER EXTENSION ----
    combined_ext, index_ext, middle_ext = get_index_middle_extension(hand_landmarks)
    
    # Use combined extension for wrist control
    # Fingers curled = wrist one direction, fingers extended = wrist opposite direction
    combined_ext = clamp(combined_ext, finger_extension_min, finger_extension_max)
    wrist_rot = map_range(combined_ext, finger_extension_min, finger_extension_max, 
                          wrist_min, wrist_max)

    # ---- GRIPPER from thumb-index pinch ----
    pinch_dist1 = ((thumb_tip.x - index_tip.x)**2 +
                   (thumb_tip.y - index_tip.y)**2 +
                   (thumb_tip.z - index_tip.z)**2)**0.5
    
    thumb_ip = hand_landmarks.landmark[3]
    pinch_dist2 = ((thumb_ip.x - index_mcp.x)**2 +
                   (thumb_ip.y - index_mcp.y)**2 +
                   (thumb_ip.z - index_mcp.z)**2)**0.5
    
    pinch_dist = pinch_dist1 * 0.7 + pinch_dist2 * 0.3
    pinch_dist = clamp(pinch_dist, pinch_dist_min, pinch_dist_max)
    gripper = map_range(pinch_dist, pinch_dist_min, pinch_dist_max, 
                        gripper_min, gripper_max)

    return [base, shoulder, elbow, wrist_rot, gripper], combined_ext, index_ext, middle_ext

def apply_speed_control(target, current, velocity, dt=1.0):
    """Apply speed limits and acceleration to joint movement"""
    new_current = current.copy()
    new_velocity = velocity.copy()
    
    for i in range(5):
        error = target[i] - current[i]
        
        if abs(error) < 0.1:
            new_velocity[i] = 0
            continue
        
        desired_vel = error * 0.1
        vel_change = desired_vel - velocity[i]
        vel_change = clamp(vel_change, -ACCELERATION, ACCELERATION)
        new_velocity[i] = velocity[i] + vel_change
        
        new_velocity[i] = clamp(new_velocity[i], -MAX_SPEED, MAX_SPEED)
        
        if abs(new_velocity[i]) < MIN_SPEED and abs(error) > dead_zone:
            new_velocity[i] = MIN_SPEED if error > 0 else -MIN_SPEED
        
        new_current[i] += new_velocity[i]
        
        if (error > 0 and new_current[i] > target[i]) or \
           (error < 0 and new_current[i] < target[i]):
            new_current[i] = target[i]
            new_velocity[i] = 0
    
    return new_current, new_velocity

def send_to_arduino(angles):
    """Send angles to Arduino with error handling"""
    if not debug and ser is not None:
        try:
            ser.write(bytearray([int(a) for a in angles]))
            return True
        except serial.SerialException as e:
            print(f"Serial write error: {e}")
            return False
    return False

# ------------------- Main Loop -------------------
with mp_hands.Hands(model_complexity=0,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as hands:
    
    frame_count = 0
    last_send_time = time.time()
    send_interval = 0.05
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty frame.")
            continue

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        current_time = time.time()
        dt = current_time - last_send_time

        # Default values
        combined_ext = 0.5
        index_ext = 0.5
        middle_ext = 0.5

        # Update target angles based on hand detection
        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 1:
            # Hand detected
            last_hand_detected_time = current_time
            
            if not hand_present:
                hand_present = True
                print("Hand detected - Starting control")
            
            hand_landmarks = results.multi_hand_landmarks[0]
            raw_target, combined_ext, index_ext, middle_ext = landmark_to_servo_angle(hand_landmarks)
            
            # Apply smoothing to target
            for i in range(5):
                target_angles[i] = smoothing_alpha * raw_target[i] + \
                                   (1 - smoothing_alpha) * target_angles[i]
            
            # Draw hand landmarks
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
            
        else:
            # No hand or multiple hands detected
            if hand_present:
                if current_time - last_hand_detected_time > NO_HAND_TIMEOUT:
                    hand_present = False
                    print("No hand detected - Returning to home")
            
            # Gradually move target to home
            for i in range(5):
                target_angles[i] += (HOME_POSITION[i] - target_angles[i]) * 0.05

        # Apply speed control to current angles
        current_angles, angle_velocities = apply_speed_control(
            target_angles, current_angles, angle_velocities, dt
        )

        # Send to Arduino at fixed intervals
        if current_time - last_send_time >= send_interval:
            # Check if movement has changed significantly
            movement = sum(abs(current_angles[i] - prev_angles[i]) for i in range(5))
            
            if movement > dead_zone or not hand_present:
                status = "ACTIVE" if hand_present else "HOME"
                print(f"{status} - B:{int(current_angles[0]):3d} S:{int(current_angles[1]):3d} "
                      f"E:{int(current_angles[2]):3d} W:{int(current_angles[3]):3d} G:{int(current_angles[4]):3d}")
                
                prev_angles = current_angles.copy()
                send_to_arduino([int(a) for a in current_angles])
            
            last_send_time = current_time

        # Display information on screen
        image = cv2.flip(image, 1)
        
        # Status
        status_color = (0, 255, 0) if hand_present else (0, 0, 255)
        status_text = "ACTIVE - Hand Controlling" if hand_present else "IDLE - Home Position"
        cv2.putText(image, status_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA)
        
        # Current angles
        cv2.putText(image, 
                    f"Base:{int(current_angles[0]):3d}  Shoulder:{int(current_angles[1]):3d}  Elbow:{int(current_angles[2]):3d}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image, 
                    f"Wrist:{int(current_angles[3]):3d}  Gripper:{int(current_angles[4]):3d}", 
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Control methods
        cv2.putText(image, "Base: Hand Left/Right | Shoulder: Hand Up/Down", (10, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 0), 1, cv2.LINE_AA)
        cv2.putText(image, "Elbow: Hand Forward/Back | Wrist: Index+Middle Extend/Curled", (10, 130), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 0), 1, cv2.LINE_AA)
        cv2.putText(image, "Gripper: Pinch Fingers", (10, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 0), 1, cv2.LINE_AA)
        
        # Show finger extension values
        if hand_present:
            cv2.putText(image, f"Index Ext: {index_ext:.2f}  Middle Ext: {middle_ext:.2f}  Combined: {combined_ext:.2f}", 
                       (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Visual indicator for wrist control
            wrist_direction = "EXTENDED" if combined_ext > 0.6 else "CURLED" if combined_ext < 0.4 else "NEUTRAL"
            wrist_color = (0, 255, 0) if combined_ext > 0.6 else (0, 0, 255) if combined_ext < 0.4 else (255, 255, 0)
            cv2.putText(image, f"Wrist: {wrist_direction}", (10, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, wrist_color, 1, cv2.LINE_AA)
        
        # Gripper status
        if hand_present:
            if current_angles[4] < 30:
                gripper_status = "CLOSED"
                gripper_color = (0, 0, 255)
            elif current_angles[4] > 150:
                gripper_status = "OPEN"
                gripper_color = (0, 255, 0)
            else:
                gripper_status = "MID"
                gripper_color = (255, 255, 0)
            
            cv2.putText(image, f"Gripper: {gripper_status}", (10, 230), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, gripper_color, 2, cv2.LINE_AA)

        cv2.imshow('5-DOF Robot - Two-Finger Wrist Control', image)

        if write_video:
            out.write(image)

        if cv2.waitKey(5) & 0xFF == 27:  # ESC key
            break

# Smooth return to home before exit
print("Returning to home position...")
steps = 50
for step in range(steps):
    for i in range(5):
        current_angles[i] += (HOME_POSITION[i] - current_angles[i]) * 0.1
    send_to_arduino([int(a) for a in current_angles])
    time.sleep(0.05)

cap.release()
if write_video:
    out.release()
cv2.destroyAllWindows()
if ser is not None:
    ser.close()
print("Program ended")