from langchain.tools import BaseTool
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
import pygame
import random
import speech_recognition as sr
import wave
import datetime
import cv2
import numpy as np


# Tool 2: Play Sound
class PlaySoundTool(BaseTool):
    name = "play_sound"
    description = """用於播放聲音的工具。
    當使用者提到「播放喇叭」、「播放聲音」、「讓喇叭發聲」等相關詞時使用。
    """
    
    def _init__(self):
        pygame.mixer.init()
        
    def _run(self):
        # 預設音效檔案清單
        sound_files = ['beep.wav', 'notification.wav', 'alert.wav']
        # 隨機選擇一個音效檔案播放
        selected_sound = random.choice(sound_files)
        pygame.mixer.Sound(selected_sound).play()
        return "已播放聲音"


# Tool 2: STT
class SpeechToTextTool(BaseTool):
    name = "speech_to_text"
    description = """用於將語音轉換為文字的工具。
    當使用者提到「STT」、「語音轉文字」等相關詞時使用。
    """
    
    def _run(self):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("請說話...")
            audio = recognizer.listen(source)
            
        # 儲存音訊檔案
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_filename = f"audio_{timestamp}.wav"
        text_filename = f"transcript_{timestamp}.txt"
        
        # 儲存音訊
        with wave.open(audio_filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(audio.get_wav_data())
            
        # 進行語音辨識
        try:
            text = recognizer.recognize_google(audio, language='zh-TW')
            # 儲存文字檔
            with open(text_filename, 'w', encoding='utf-8') as f:
                f.write(text)
            return f"語音轉換完成。\n音訊檔案：{audio_filename}\n文字檔案：{text_filename}\n辨識內容：{text}"
        except sr.UnknownValueError:
            return "無法辨識語音內容"

# Tool 3: Count People
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


#  initialization of agent
def setup_agent():
    # 初始化 ChatGPT
    llm = ChatOpenAI(
        temperature=0,
        model_name="gpt-3.5-turbo"
    )
    
    # 初始化所有 tools
    tools = [
        PlaySoundTool(),
        SpeechToTextTool(),
        PeopleCounterTool()
    ]
    
    # 設定記憶體
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    # 初始化 agent
    agent = initialize_agent(
        tools,
        llm,
        agent="chat-conversational-react-description",
        memory=memory,
        verbose=True
    )
    
    return agent


def main():
    agent = setup_agent()
    
    print("AI Assistant 已啟動，請說出您的需求...")
    
    while True:
        user_input = input("您: ")
        if user_input.lower() in ['退出', 'exit', 'quit']:
            break
            
        try:
            response = agent.run(user_input)
            print(f"AI: {response}")
        except Exception as e:
            print(f"發生錯誤: {str(e)}")
            
if __name__ == "__main__":
    main()
