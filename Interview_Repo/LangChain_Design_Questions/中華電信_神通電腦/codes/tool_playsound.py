"""
程式描述：
play sound
"""

from langchain.tools import BaseTool
import pygame
import random

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
