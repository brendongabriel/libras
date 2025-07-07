from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam

# Configurações
input_shape = (224, 224, 3)
num_classes = 26  # A-Z

# Carregar MobileNetV2 pré-treinada (sem o topo)
base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=input_shape)

# Congelar convoluções
for layer in base_model.layers:
    layer.trainable = False

# Cabeçalho personalizado
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(num_classes, activation='softmax')(x)

# Modelo final
model = Model(inputs=base_model.input, outputs=predictions)

# Compilar
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Salvar (opcional, antes do treino)
model.save("mobilenetv2_libras.h5")
