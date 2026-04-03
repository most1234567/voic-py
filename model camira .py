import cv2
from ultralytics import YOLO

# تحميل موديل الـ Segmentation (النسخة النانو)
model_seg = YOLO('yolov8n-seg.pt')

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # 1. تحليل المشهد (Segmentation)
    results = model_seg(frame, conf=0.5, verbose=False)
    
    # 2. استخراج أسماء العناصر الفريدة في المشهد بالكامل
    unique_elements = set()
    for r in results:
        if r.boxes:
            for cls in r.boxes.cls:
                unique_elements.add(model_seg.names[int(cls)])
    
    # 3. بناء وصف عام للمشهد للـ LLM
    if unique_elements:
        scene_summary = ", ".join(unique_elements)
        scene_prompt = f"The current scene contains a segmented view of: {scene_summary}. Describe the general environment and the relationship between these elements."
        
        # طباعة الـ Prompt في الـ Terminal للتأكد من المحتوى
        # يمكنك هنا استدعاء API الـ LLM (مثل OpenAI)
        print("-" * 30)
        print(f"Scene Description: {scene_summary}")
        print(f"LLM Prompt: {scene_prompt}")

    # عرض المشهد مع تلوين الأجسام (Masks)
    cv2.imshow('Task 2: Scene Segmentation (Press Q to Quit)', results[0].plot())

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()

import cv2
from ultralytics import YOLO

# 1. تحميل الموديلات (تأكد من وجود إنترنت لأول مرة لتحميلهم تلقائياً)
model_det = YOLO('yolov8n.pt')      # للمهمة الأولى (Recognition)
model_seg = YOLO('yolov8n-seg.pt')  # للمهمة الثانية (Segmentation)

def start_camera_test():
    # فتح كاميرا اللابتوب (رقم 0 هو الافتراضي)
    cap = cv2.VideoCapture(0)
    
    print("اضغط 'q' للخروج من الكاميرا...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # --- تجربة Task 1 (Recognition) ---
        results_det = model_det(frame, conf=0.5, verbose=False)
        
        # تجميع الأشياء للـ Prompt
        detected_names = [model_det.names[int(box.cls)] for r in results_det for box in r.boxes]
        
        if detected_names:
            prompt_task1 = f"I see {', '.join(set(detected_names))}. Give me an educational tip."
            # سنكتفي بطباعته في الـ Terminal للتأكد
            print(f"Task 1 Prompt: {prompt_task1}")

        # --- تجربة Task 2 (Segmentation) ---
        results_seg = model_seg(frame, conf=0.5, verbose=False)
        
        # عرض النتائج مرئياً على الشاشة (الرسم على الـ frame)
        annotated_frame = results_seg[0].plot() # هذا سيرسم الـ Masks والـ Boxes

        # عرض النافذة
        cv2.imshow('YOLOv8 Test - Press Q to Exit', annotated_frame)

        # الخروج عند الضغط على حرف q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_camera_test()