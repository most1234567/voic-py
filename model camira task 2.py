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