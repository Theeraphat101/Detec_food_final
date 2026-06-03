import cv2
import torch
import numpy as np
from torchvision import models, transforms

def load_model(model_path="thai_food_modelv8.pth"):
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(model.last_channel, 10)
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)
        model.eval()
        print("โหลดโมเดลสำเร็จ")
    except FileNotFoundError:
        print("ไม่พบไฟล์โมเดล:", model_path)
        exit()
    except Exception as e:
        print(f"ข้อผิดพลาดในการโหลดโมเดล001: {e}")
        exit()
    return model

thai_food_labels = ['Green_Curry', 'Khao_phat', 'Khao_soi', 'Massaman_Curry', 'Pad_Thai', 'Phanaeng_Curry', 'Phat_kaphrao', 'Roti_canai', 'Tom_kha_gai', 'Tom_yum']
thai_food_calories = {
    'Green_Curry': 240, 
    'Khao_phat': 320, 
    'Khao_soi': 290, 
    'Massaman_Curry': 250, 
    'Pad_Thai': 150, 
    'Phanaeng_Curry': 450, 
    'Phat_kaphrao': 200, 
    'Roti_canai': 350, 
    'Tom_kha_gai': 120, 
    'Tom_yum': 190
}

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def classify_food(frame, model):
    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = transform(frame_rgb).unsqueeze(0)
        outputs = model(input_tensor)
        _, predicted = outputs.max(1)
        food_name = thai_food_labels[predicted.item()]
        calories = thai_food_calories.get(food_name, "Unknown")
    except Exception as e:
        print(f"Error in classify_food: {e}")
        food_name, calories = "No food detected", "N/A"
    return food_name, calories

def main():
    model = load_model()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ไม่สามารถเปิดกล้องได้")
        return

    # กำหนดขนาดกรอบตรวจจับ (อัตราส่วนจากขนาดของเฟรม)
    x_ratio, y_ratio, w_ratio, h_ratio = 0.25, 0.25, 0.5, 0.5  # x, y, ความกว้าง, ความสูงเป็นสัดส่วน

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("ไม่สามารถอ่านเฟรมจากกล้องได้")
                break

            height, width = frame.shape[:2]

            # คำนวณพิกัดกรอบตรวจจับ
            x = int(width * x_ratio)
            y = int(height * y_ratio)
            w = int(width * w_ratio)
            h = int(height * h_ratio)

            # วาดกรอบคงที่ (Static Bounding Box)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # กรอบสีเขียว

            # ครอบภาพเฉพาะส่วนในกรอบเพื่อนำไปตรวจจับ
            cropped_frame = frame[y:y + h, x:x + w]
            food_name, calories = classify_food(cropped_frame, model)

            # แสดงข้อมูลชื่ออาหารและแคลอรี่
            cv2.putText(frame, f"Food: {food_name}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Calories: {calories} kcal", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("Thai Food Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("ออกจากโปรแกรม")
                break

    except Exception as e:
        print(f"ข้อผิดพลาดในโปรแกรมหลัก: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()