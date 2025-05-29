import matplotlib
matplotlib.use("Agg")
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense, Input, concatenate
from tensorflow.keras import backend as K
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import os
import glob
import seaborn as sns

class LivenessNetV3:
    @staticmethod
    def build(width, height, depth, classes, dropout_rate=0.3):
        inputShape = (height, width, depth)
        chanDim = -1
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        input_tensor = Input(shape=inputShape)
        x = Conv2D(64, (3, 3), padding="same")(input_tensor)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = Conv2D(64, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(dropout_rate)(x)

        x = Conv2D(128, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = Conv2D(128, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(dropout_rate)(x)

        x = Conv2D(256, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = Conv2D(256, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(dropout_rate)(x)

        return Model(inputs=input_tensor, outputs=x)

def frequency_analysis_features(image):
    """Trích xuất đặc trưng từ phân tích tần số (trên kênh Y của YCrCb)."""
    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    y_channel = ycrcb[:,:,0]
    f_y = np.fft.fft2(y_channel)
    fshift_y = np.fft.fftshift(f_y)
    magnitude_spectrum_y = np.log(np.abs(fshift_y) + 1)
    mean_mag = np.mean(magnitude_spectrum_y)
    std_mag = np.std(magnitude_spectrum_y)
    max_mag = np.max(magnitude_spectrum_y)
    return [mean_mag, std_mag, max_mag]

# Configuration parameters
dataset_path = "C:/users/huydo/appdata/local/programs/python/python311/lib/site-packages/face_anti_spoofing/dataset"
model_path = "C:/users/huydo/appdata/local/programs/python/python311/lib/site-packages/face_anti_spoofing/liveness_v3_freq.h5"
label_encoder_path = "C:/users/huydo/appdata/local/programs/python/python311/lib/site-packages/face_anti_spoofing/le.pickle"
plot_path = "C:/users/huydo/appdata/local/programs/python/python311/lib/site-packages/face_anti_spoofing/plot_v3_freq.png"
confusion_matrix_path = "C:/users/huydo/appdata/local/programs/python/python311/lib/site-packages/face_anti_spoofing/confusion_matrix_v3_freq.png"
scaler_path = "C:/users/huydo/appdata/local/programs/python/python311/lib/site-packages/face_anti_spoofing/scaler_freq.pkl"

# Training parameters
INIT_LR = 1e-3
BS = 64
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
REDUCE_LR_PATIENCE = 7
DROPOUT_RATE = 0.45
IMAGE_SIZE = 48

# Data augmentation
aug_train = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.25,
    width_shift_range=0.25,
    height_shift_range=0.25,
    shear_range=0.2,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    vertical_flip=False,
    fill_mode="nearest"
)

aug_val = ImageDataGenerator(rescale=1./255)

# Load and preprocess images
print("[INFO] loading images...")
imagePaths = list(paths.list_images(dataset_path))
pixel_data = []
labels = []
frequency_features_list = []

for imagePath in imagePaths:
    label = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    pixel_data.append(image)
    labels.append(label)
    freq_feats = frequency_analysis_features(image)
    frequency_features_list.append(freq_feats)

# Normalize pixel values
pixel_data = np.array(pixel_data, dtype="float")
frequency_features_array = np.array(frequency_features_list)

# Encode labels
le = LabelEncoder()
labels = le.fit_transform(labels)
labels = to_categorical(labels, 2)

# Split data
(trainX_pixel, testX_pixel, trainY, testY, trainX_freq, testX_freq) = train_test_split(
    pixel_data, labels, frequency_features_array, test_size=0.2, random_state=42, stratify=labels
)

# Chuẩn hóa đặc trưng tần số
scaler_freq = StandardScaler()
trainX_freq_scaled = scaler_freq.fit_transform(trainX_freq)
testX_freq_scaled = scaler_freq.transform(testX_freq)

# Lưu scaler
with open(scaler_path, "wb") as f:
    pickle.dump(scaler_freq, f)
print(f"[INFO] Saved frequency feature scaler to: {scaler_path}")

# Build combined model
print("[INFO] building combined model...")
base_model = LivenessNetV3.build(width=IMAGE_SIZE, height=IMAGE_SIZE, depth=3, classes=len(le.classes_), dropout_rate=DROPOUT_RATE)
flatten_base = Flatten()(base_model.output)

input_freq = Input(shape=(trainX_freq_scaled.shape[1],))
dense_freq = Dense(64, activation="relu")(input_freq)

merged = concatenate([flatten_base, dense_freq])
dense_merged = Dense(256, activation="relu")(merged)
output = Dense(len(le.classes_), activation="softmax")(dense_merged)

model_combined = Model(inputs=[base_model.input, input_freq], outputs=output)
model_combined.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=INIT_LR), metrics=["accuracy"])

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=REDUCE_LR_PATIENCE, min_lr=1e-6)
model_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_accuracy', save_best_only=True, verbose=1)
callbacks = [early_stopping, reduce_lr, model_checkpoint]

# Hàm generator kết hợp dữ liệu ảnh và tần số
def combined_generator(image_data, image_labels, frequency_features, batch_size, image_data_generator, shuffle=False):
    gen = image_data_generator.flow(image_data, image_labels, batch_size=batch_size, shuffle=shuffle)
    freq_index = 0
    while True:
        batch_images, batch_labels = next(gen)
        start_index = freq_index * batch_size
        end_index = (freq_index + 1) * batch_size
        batch_freq_features = frequency_features[start_index:end_index]
        yield (batch_images, batch_freq_features), batch_labels  # Trả về tuple cho features
        freq_index = (freq_index + 1) % (len(frequency_features) // batch_size + (1 if len(frequency_features) % batch_size != 0 else 0))

# Tạo ImageDataGenerator instances (giữ nguyên)
train_image_generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.25,
    width_shift_range=0.25,
    height_shift_range=0.25,
    shear_range=0.2,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    vertical_flip=False,
    fill_mode="nearest"
)

val_image_generator = ImageDataGenerator(rescale=1./255)

# Tạo generators cho huấn luyện và validation (giữ nguyên)
train_generator = combined_generator(trainX_pixel, trainY, trainX_freq_scaled, BS, train_image_generator, shuffle=True)
val_generator = combined_generator(testX_pixel, testY, testX_freq_scaled, BS, val_image_generator, shuffle=False)

# Tính số lượng steps cho mỗi epoch (giữ nguyên)
train_steps = len(trainX_pixel) // BS
val_steps = len(testX_pixel) // BS

# Huấn luyện mô hình kết hợp (giữ nguyên)
print("[INFO] training combined network for {} epochs...".format(EPOCHS))
H = model_combined.fit(
    train_generator,
    steps_per_epoch=train_steps,
    validation_data=val_generator,
    validation_steps=val_steps,
    epochs=EPOCHS,
    callbacks=callbacks
    
)


# Evaluate combined model
print("[INFO] evaluating combined network...")
best_model_combined = tf.keras.models.load_model(model_path)
predictions = best_model_combined.predict([testX_pixel, testX_freq_scaled], batch_size=BS)
print(classification_report(
    testY.argmax(axis=1),
    predictions.argmax(axis=1),
    target_names=le.classes_
))

# Generate and save confusion matrix
cm = confusion_matrix(testY.argmax(axis=1), predictions.argmax(axis=1))
plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix (Combined Model)')
plt.savefig(confusion_matrix_path)
print(f"[INFO] Confusion matrix saved to: {confusion_matrix_path}")

# Save the label encoder
try:
    with open(label_encoder_path, "wb") as f:
        pickle.dump(le, f)
    print(f"[INFO] Saved label encoder to: {label_encoder_path}")
except Exception as e:
    print(f"[ERROR] Could not save label encoder: {e}")

# Plot training history
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, len(H.history["loss"])), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, len(H.history["val_loss"])), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, len(H.history["accuracy"])), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, len(H.history["val_accuracy"])), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy (Combined Model)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(plot_path)
print(f"[INFO] Training plot saved to: {plot_path}")

print("[INFO] TRAINING COMPLETE (WITH FREQUENCY ANALYSIS)!")