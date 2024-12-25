from langchain.tools import BaseTool
import cv2
import numpy as np

class PeopleCounterTool(BaseTool):
    name = "count_people"
    description = """用於計算攝影機畫面中的人數。
    當使用者提到「現場人數」、「鏡頭中人數」等相關詞時使用。
    """
    
    def _run(self):
        # 初始化攝影機
        cap = cv2.VideoCapture(0)
        
        # 載入預訓練的人臉偵測器
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # 捕捉一幀影像
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return "無法取得攝影機畫面"
            
        # 轉換為灰階影像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 偵測人臉
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5
        )
        
        # 釋放攝影機
        cap.release()
        
        return f"目前畫面中偵測到 {len(faces)} 人"
