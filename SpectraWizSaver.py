import time

import pyautogui
import re


def save():
    # Programm should be in focus
    pyautogui.keyDown("lalt")
    time.slep(0.3)
    pyautogui.press("f")
    time.slep(0.3)
    pyautogui.press("s")
    time.slep(0.3)
    pyautogui.press("s")

    time.slep(0.3)
    pyautogui.press("a")
    time.slep(0.3)
    pyautogui.press("enter")
    time.slep(0.5)
    pyautogui.press("f")