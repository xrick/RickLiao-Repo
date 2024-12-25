from langchain.tools import BaseTool
import speech_recognition as sr
import wave
import datetime

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