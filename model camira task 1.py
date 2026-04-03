import cv2
from ultralytics import YOLO
import pyttsx3

# إعداد الموديل والصوت
model = YOLO('yolov8n.pt')
engine = pyttsx3.init()

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # 1. اكتشاف الأشياء بـ threshold 0.5
    results = model(frame, conf=0.5, verbose=False)
    
    detected_items = []
    for r in results:
        for box in r.boxes:
            name = model.names[int(box.cls)]
            conf = float(box.conf)
            detected_items.append(f"{name} ({conf:.2f})")

    # 2. بناء Prompt واحد لكل الأشياء (عند الضغط على مفتاح 's' مثلاً للتقليل من الإزعاج)
    if detected_items and cv2.waitKey(1) & 0xFF == ord('s'):
        all_objects = ", ".join(list(set(detected_items))) # حذف التكرار
        prompt = f"I can see these objects: {all_objects}. Tell me an educational fact about them."
        
        print(f"Generated Prompt: {prompt}")
        
        # 3. تحويل الرد لصوت (محاكاة رد الـ LLM)
        engine.say(f"I found {len(detected_items)} objects. Let's learn about them.")
        engine.runAndWait()

    # عرض الكاميرا مع المربعات
    cv2.imshow('Task 1: Object Recognition (Press S for TTS, Q to Quit)', results[0].plot())
    
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()