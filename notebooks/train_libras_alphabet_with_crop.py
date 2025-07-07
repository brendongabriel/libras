
# LIBRAS Alphabet Recognition - Treinamento com Recorte de Mão (via XML)

# Etapa 1: Imports e Configuração
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam

# Etapa 2: Função para carregar imagem e aplicar bounding box
def load_and_crop(xml_file, image_file, target_size=(224, 224)):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    bbox = root.find("object").find("bndbox")
    xmin = int(bbox.find("xmin").text)
    ymin = int(bbox.find("ymin").text)
    xmax = int(bbox.find("xmax").text)
    ymax = int(bbox.find("ymax").text)

    img = cv2.imread(image_file)
    if img is None:
        return None

    cropped = img[ymin:ymax, xmin:xmax]
    resized = cv2.resize(cropped, target_size)
    return resized

# Etapa 3: Carregar dataset
data_dir = "../dataset"  # ajuste conforme o local do seu dataset
images = []
labels = []
classes = sorted(os.listdir(data_dir))
class_to_idx = {c: i for i, c in enumerate(classes)}

for class_name in classes:
    img_files = glob(os.path.join(data_dir, class_name, "*.jpg"))
    for img_file in tqdm(img_files, desc=f"Processando {class_name}"):
        xml_file = img_file.replace(".jpg", ".xml")
        if os.path.exists(xml_file):
            img = load_and_crop(xml_file, img_file)
            if img is not None:
                images.append(img)
                labels.append(class_to_idx[class_name])

X = np.array(images, dtype='float32') / 255.0
y = to_categorical(np.array(labels), num_classes=len(classes))

# Etapa 4: Dividir treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Etapa 5: Criar modelo
base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(len(classes), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Etapa 6: Treinar modelo
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

# Etapa 7: Salvar modelo
model.save("mobilenetv2_libras_recorte.h5")

# Etapa 8: Visualizar resultados
plt.plot(history.history['accuracy'], label='acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.title("Acurácia")
plt.legend()
plt.show()
