import pyautogui
import pydirectinput
import time
import win32con

from directkeys import (PressKey, key_press, ReleaseKey)
import win32api

time.sleep(2)
pydirectinput.press('i')
time.sleep(1)
pyautogui.moveTo(776, 471, duration=0.2)
time.sleep(0.5)
pydirectinput.rightClick(776, 471)
time.sleep(1)
# pyautogui.rightClick(x=868, y=471)
# pydirectinput.press('a')  # Simulate pressing the Escape key.
key_press('a')
# PressKey(0x1E)
# time.sleep(0.05)
# ReleaseKey(0x1E)
print('打完了')
