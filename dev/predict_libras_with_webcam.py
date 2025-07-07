
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Carregar modelo treinado
model = load_model("../models/mobilenetv2_libras_recorte.h5")

# Mapear índice para letra
labels = [chr(i) for i in range(65, 91)]  # A-Z

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erro ao abrir webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            # Obter bounding box da mão
            x_list = [lm.x for lm in handLms.landmark]
            y_list = [lm.y for lm in handLms.landmark]
            xmin = int(min(x_list) * w) - 20
            ymin = int(min(y_list) * h) - 20
            xmax = int(max(x_list) * w) + 20
            ymax = int(max(y_list) * h) + 20

            # Garantir que está dentro do frame
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(w, xmax)
            ymax = min(h, ymax)

            hand_img = frame[ymin:ymax, xmin:xmax]
            if hand_img.size == 0:
                continue

            # Preprocessar
            hand_img = cv2.resize(hand_img, (224, 224))
            hand_img = preprocess_input(hand_img)
            hand_img = np.expand_dims(hand_img, axis=0)

            # Predição
            pred = model.predict(hand_img)
            idx = np.argmax(pred)
            label = labels[idx]

            # Mostrar resultado
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
            cv2.putText(frame, label, (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (255,0,0), 2)

            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("LIBRAS Recognizer", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Esc para sair
        break

cap.release()
cv2.destroyAllWindows()
