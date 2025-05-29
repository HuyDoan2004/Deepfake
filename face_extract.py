import numpy as np
import cv2
import os

# Tham số đầu vào
input_video_real = "C:/users/huydo/appdata/local/programs/python/python311/lib/site-packages/face_anti_spoofing/video/real.mp4"
input_video_fake = "C:/users/huydo/appdata/local/programs/python/python311/lib/site-packages/face_anti_spoofing/video/fake.mp4"
output_directory_real = "C:/users/huydo/appdata/local/programs/python/python311/lib/site-packages/face_anti_spoofing/dataset/real"
output_directory_fake = "C:/users/huydo/appdata/local/programs/python/python311/lib/site-packages/face_anti_spoofing/dataset/fake"
detector_path = "C://Users//huydo//AppData//Local//Programs//Python//Python311//Lib//site-packages//face_anti_spoofing//face_detector"
confidence_threshold = 0.5
max_images_real = 10000
max_images_fake = 10000

# Load model SSD nhận diện mặt
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([detector_path, "deploy.prototxt"])
modelPath = os.path.sep.join([detector_path, "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# Hàm xử lý video, cắt ảnh phân bố đều
def process_video(video_path, output_dir, max_images):
    vs = cv2.VideoCapture(video_path)
    total_frames = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print(f"[ERROR] Không lấy được tổng frame video {video_path}")
        return

    frame_step = max(1, total_frames // max_images)
    print(f"[INFO] Video: {video_path} | Tổng frame: {total_frames} | Lấy mẫu mỗi {frame_step} frame")

    saved = 0
    frame_number = 0
    while True:
        grabbed, frame = vs.read()
        if not grabbed:
            break

        # Lấy mẫu phân bố đều
        if frame_number % frame_step == 0:
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300),
                                        (104.0, 177.0, 123.0))
            net.setInput(blob)
            detections = net.forward()

            if len(detections) > 0:
                i = np.argmax(detections[0, 0, :, 2])
                confidence_score = detections[0, 0, i, 2]

                if confidence_score > confidence_threshold:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # Giới hạn trong ảnh gốc
                    startX, startY = max(0, startX), max(0, startY)
                    endX, endY = min(w - 1, endX), min(h - 1, endY)

                    if endX > startX and endY > startY:
                        face = frame[startY:endY, startX:endX]
                        if face.size > 0:
                            file_path = os.path.join(output_dir, f"{os.path.basename(output_dir)}_{saved}.png")
                            cv2.imwrite(file_path, face)
                            saved += 1
                            print(f"[INFO] saved {file_path} with confidence {confidence_score:.3f}")

                            if saved >= max_images:
                                print(f"[INFO] Đã lưu đủ {max_images} ảnh, dừng.")
                                break

        frame_number += 1

    vs.release()
    return saved

# Tạo thư mục nếu chưa có
os.makedirs(output_directory_real, exist_ok=True)
os.makedirs(output_directory_fake, exist_ok=True)

# Xử lý video real và fake
print("[INFO] Processing REAL video...")
saved_real = process_video(input_video_real, output_directory_real, max_images_real)

print("[INFO] Processing FAKE video...")
saved_fake = process_video(input_video_fake, output_directory_fake, max_images_fake)

print(f"[INFO] Tổng ảnh real đã lưu: {saved_real}")
print(f"[INFO] Tổng ảnh fake đã lưu: {saved_fake}")

cv2.destroyAllWindows()
