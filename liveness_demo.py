from imutils.video import VideoStream
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import imutils
import pickle
import time
import cv2
import os
from sklearn.preprocessing import StandardScaler

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;0"

# Định nghĩa đường dẫn mô hình và label encoder
model_path = "C:/users/huydo/appdata/local/programs/python/python311/lib/site-packages/face_anti_spoofing/liveness_v3_freq.h5"  # Đảm bảo là .h5
le_path = "C:/users/huydo/appdata/local/programs/python/python311/lib/site-packages/face_anti_spoofing/le.pickle"
detector_path = "C:/Users/huydo/AppData/Local/Programs/Python/Python311/Lib/site-packages/face_anti_spoofing/face_detector"
scaler_path = "C:/users/huydo/appdata/local/programs/python/python311/lib/site-packages/face_anti_spoofing/scaler_freq.pkl" # Đường dẫn đến scaler đã lưu

# Load model nhận diện khuôn mặt
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([detector_path, "deploy.prototxt"])
modelPath = os.path.sep.join([detector_path, "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# Load mô hình nhận diện fake/real
print("[INFO] loading liveness detector...")
model = load_model(model_path)  # Load mô hình .h5
le = pickle.loads(open(le_path, "rb").read())
scaler = pickle.loads(open(scaler_path, "rb").read()) # Load scaler

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
    return np.array([[mean_mag, std_mag, max_mag]]) # Trả về mảng 2D để phù hợp với scaler

# Đọc video từ webcam
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

while True:
    # Đọc ảnh từ webcam
    frame = vs.read()
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)

            # Lấy vùng khuôn mặt
            face = frame[startY:endY, startX:endX]
            face_resized = cv2.resize(face, (48, 48))
            face_normalized = face_resized.astype("float") / 255.0
            face_array = img_to_array(face_normalized)
            face_expanded = np.expand_dims(face_array, axis=0)

            # Trích xuất đặc trưng tần số từ vùng khuôn mặt
            freq_features = frequency_analysis_features(face)
            freq_features_scaled = scaler.transform(freq_features)

            # Dự đoán fake/real (cung cấp cả dữ liệu pixel và đặc trưng tần số)
            preds = model.predict([face_expanded, freq_features_scaled])[0]

            j = np.argmax(preds)
            label = le.classes_[j]
            label_text = "{}: {:.4f}".format(label, preds[j])

            if (j == 0):
                cv2.putText(frame, label_text, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                    (0, 0, 255), 2)
            else:
                cv2.putText(frame, label_text, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                    (0, 255, 0), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Cleanup
cv2.destroyAllWindows()
vs.stop()