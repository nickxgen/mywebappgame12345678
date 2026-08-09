import cv2
import time
import numpy as np
import mediapipe as mp
from pynput.keyboard import Controller, Key

# Initialize Keyboard Controller
keyboard = Controller()

# Initialize MediaPipe Hands for 2 Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    model_complexity=0,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# Key Layout Configuration
KEYS = [
    ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
    ["A", "S", "D", "F", "G", "H", "J", "K", "L"],
    ["Z", "X", "C", "V", "B", "N", "M"],
    ["SPACE", "BACKSPACE", "CLEAR"]
]

class KeyButton:
    def __init__(self, x, y, w, h, text):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.text = text
        self.is_hovered = False
        self.is_pressed = False
        self.hover_start_time = None

def build_keyboard(start_x=30, start_y=160, key_size=50, gap=10):
    buttons = []
    for row_idx, row in enumerate(KEYS[:3]):
        offset = row_idx * 20
        for col_idx, key in enumerate(row):
            x = start_x + offset + col_idx * (key_size + gap)
            y = start_y + row_idx * (key_size + gap)
            buttons.append(KeyButton(x, y, key_size, key_size, key))

    special_y = start_y + 3 * (key_size + gap)
    buttons.append(KeyButton(start_x + 80, special_y, 220, key_size, "SPACE"))
    buttons.append(KeyButton(start_x + 310, special_y, 140, key_size, "BACKSPACE"))
    buttons.append(KeyButton(start_x + 460, special_y, 100, key_size, "CLEAR"))

    return buttons

buttons = build_keyboard()
typed_text = ""

# Dwell Activation Settings (In Seconds)
DWELL_TIME = 0.12  # 120ms dwell delay for fast typing

# Tracks state across both hands: {hand_id: {"last_key": str, "cooldown_until": float}}
hand_states = {}

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    current_time = time.time()
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    cursors = []

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract Index Tip landmark (8)
            idx_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            ix, iy = int(idx_tip.x * w), int(idx_tip.y * h)
            cursors.append((idx, ix, iy))

            # Draw pointer cursor per hand
            cv2.circle(frame, (ix, iy), 9, (255, 0, 255), -1)

    # Reset hover states
    for btn in buttons:
        btn.is_hovered = False
        btn.is_pressed = False

    # Process interaction for each detected hand
    for hand_id, cx, cy in cursors:
        if hand_id not in hand_states:
            hand_states[hand_id] = {"hover_key": None, "hover_start": 0, "cooldown": 0}

        state = hand_states[hand_id]

        # Check which button the finger is over
        hovered_btn = None
        for btn in buttons:
            if btn.x < cx < btn.x + btn.w and btn.y < cy < btn.y + btn.h:
                hovered_btn = btn
                btn.is_hovered = True
                break

        # Dwell Logic per hand
        if hovered_btn:
            if state["hover_key"] == hovered_btn.text:
                dwell_elapsed = current_time - state["hover_start"]
                
                # Check dwell threshold and hand cooldown
                if dwell_elapsed >= DWELL_TIME and current_time > state["cooldown"]:
                    hovered_btn.is_pressed = True
                    
                    # Execute Type Action
                    if hovered_btn.text == "SPACE":
                        typed_text += " "
                        keyboard.press(Key.space)
                        keyboard.release(Key.space)
                    elif hovered_btn.text == "BACKSPACE":
                        typed_text = typed_text[:-1]
                        keyboard.press(Key.backspace)
                        keyboard.release(Key.backspace)
                    elif hovered_btn.text == "CLEAR":
                        typed_text = ""
                    else:
                        typed_text += hovered_btn.text
                        keyboard.press(hovered_btn.text.lower())
                        keyboard.release(hovered_btn.text.lower())

                    # Set cooldown (0.25s) before this same hand can trigger another key
                    state["cooldown"] = current_time + 0.25
                    state["hover_key"] = None
            else:
                state["hover_key"] = hovered_btn.text
                state["hover_start"] = current_time
        else:
            state["hover_key"] = None

    # --- UI RENDERING ---
    
    # 1. Output Text Box
    cv2.rectangle(frame, (30, 20), (w - 30, 90), (30, 30, 30), -1)
    cv2.rectangle(frame, (30, 20), (w - 30, 90), (200, 200, 200), 2)
    cv2.putText(frame, typed_text[-35:], (45, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

    # 2. Keyboard Buttons
    for btn in buttons:
        color = (0, 255, 0) if btn.is_pressed else ((255, 200, 0) if btn.is_hovered else (60, 60, 60))

        overlay = frame.copy()
        cv2.rectangle(overlay, (btn.x, btn.y), (btn.x + btn.w, btn.y + btn.h), color, -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.rectangle(frame, (btn.x, btn.y), (btn.x + btn.w, btn.y + btn.h), (255, 255, 255), 1)

        # Draw Label Text
        font_scale = 0.6 if len(btn.text) > 1 else 0.8
        text_size = cv2.getTextSize(btn.text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
        tx = btn.x + (btn.w - text_size[0]) // 2
        ty = btn.y + (btn.h + text_size[1]) // 2
        cv2.putText(frame, btn.text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)

    cv2.putText(frame, "Use both hands: Hover index finger briefly over key to type instantly", 
                (30, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("Fast 2-Hand Air Keyboard", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()