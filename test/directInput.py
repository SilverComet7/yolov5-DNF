import pyautogui
import pydirectinput
import time
import win32con

from directkeys import (PressKey, key_press, ReleaseKey)
import win32api

time.sleep(2)
pydirectinput.press('i')i
time.sleep(1)
pyautogui.moveTo(868, 471)
# pydirectinput.moveTo(868, 471, duration=1)  # Move the mouse to the x, y coordinates 100, 150.11
time.sleep(0.5)
# 鼠标在当前所在位置按下右键（只是按下，不松开）
win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
time.sleep(0.5)
# 鼠标在当前所在位置抬起右键（松开）
win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)
# pydirectinput.click(x=868, y=471, button='right')
# pydirectinput.rightClick(x=868, y=471, interval=0.1, duration=0.1)
time.sleep(1)
# pyautogui.rightClick(x=868, y=471)
# pydirectinput.press('a')  # Simulate pressing the Escape key.
key_press('a')
# PressKey(0x1E)
# time.sleep(0.05)
# ReleaseKey(0x1E)
print('打完了')
