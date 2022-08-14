import time

from directkeys import PressKey, ReleaseKey

direct_dic = {"UP": 0xC8, "DOWN": 0xD0, "LEFT": 0xCB, "RIGHT": 0xCD}

def move(direct, material=False, action_cache=None, press_delay=0.1, release_delay=0.1):
    if direct == "RIGHT":
        if action_cache != None:
            if action_cache != "RIGHT":
                if action_cache not in ["LEFT", "RIGHT", "UP", "DOWN"]:
                    ReleaseKey(direct_dic[action_cache.strip().split("_")[0]])
                    ReleaseKey(direct_dic[action_cache.strip().split("_")[1]])
                else:
                    ReleaseKey(direct_dic[action_cache])
                PressKey(direct_dic["RIGHT"])
                if not material:
                    time.sleep(press_delay)
                    ReleaseKey(direct_dic["RIGHT"])
                    time.sleep(release_delay)
                    PressKey(direct_dic["RIGHT"])
                action_cache = "RIGHT"
                print("向右移动")
            else:
                print("向右移动")
        else:
            PressKey(direct_dic["RIGHT"])
            if not material:
                time.sleep(press_delay)
                ReleaseKey(direct_dic["RIGHT"])
                time.sleep(release_delay)
                PressKey(direct_dic["RIGHT"])
            action_cache = "RIGHT"
            print("向右移动")
        return action_cache

    elif direct == "LEFT":
        if action_cache != None:
            if action_cache != "LEFT":
                if action_cache not in ["LEFT", "RIGHT", "UP", "DOWN"]:
                    ReleaseKey(direct_dic[action_cache.strip().split("_")[0]])
                    ReleaseKey(direct_dic[action_cache.strip().split("_")[1]])
                else:
                    ReleaseKey(direct_dic[action_cache])
                PressKey(direct_dic["LEFT"])
                if not material:
                    time.sleep(press_delay)
                    ReleaseKey(direct_dic["LEFT"])
                    time.sleep(release_delay)
                    PressKey(direct_dic["LEFT"])
                action_cache = "LEFT"
                print("向左移动")
            else:
                print("向左移动")
        else:
            PressKey(direct_dic["LEFT"])
            if not material:
                time.sleep(press_delay)
                ReleaseKey(direct_dic["LEFT"])
                time.sleep(release_delay)
                PressKey(direct_dic["LEFT"])
            action_cache = "LEFT"
            print("向左移动")
        return action_cache

    elif direct == "UP":
        if action_cache != None:
            if action_cache != "UP":
                if action_cache not in ["LEFT", "RIGHT", "UP", "DOWN"]:
                    ReleaseKey(direct_dic[action_cache.strip().split("_")[0]])
                    ReleaseKey(direct_dic[action_cache.strip().split("_")[1]])
                else:
                    ReleaseKey(direct_dic[action_cache])
                PressKey(direct_dic["UP"])
                # time.sleep(press_delay)
                # ReleaseKey(direct_dic["UP"])
                # time.sleep(release_delay)
                # PressKey(direct_dic["UP"])
                action_cache = "UP"
                print("向上移动")
            else:
                print("向上移动")
        else:
            PressKey(direct_dic["UP"])
            # time.sleep(press_delay)
            # ReleaseKey(direct_dic["UP"])
            # time.sleep(release_delay)
            # PressKey(direct_dic["UP"])
            action_cache = "UP"
            print("向上移动")
        return action_cache

    elif direct == "DOWN":
        if action_cache != None:
            if action_cache != "DOWN":
                if action_cache not in ["LEFT", "RIGHT", "UP", "DOWN"]:
                    ReleaseKey(direct_dic[action_cache.strip().split("_")[0]])
                    ReleaseKey(direct_dic[action_cache.strip().split("_")[1]])
                else:
                    ReleaseKey(direct_dic[action_cache])
                PressKey(direct_dic["DOWN"])
                # time.sleep(press_delay)
                # ReleaseKey(direct_dic["DOWN"])
                # time.sleep(release_delay)
                # PressKey(direct_dic["DOWN"])
                action_cache = "DOWN"
                print("向下移动")
            else:
                print("向下移动")
        else:
            PressKey(direct_dic["DOWN"])
            # time.sleep(press_delay)
            # ReleaseKey(direct_dic["DOWN"])
            # time.sleep(release_delay)
            # PressKey(direct_dic["DOWN"])
            action_cache = "DOWN"
            print("向下移动")
        return action_cache

    elif direct == "RIGHT_UP":
        if action_cache != None:
            if action_cache != "RIGHT_UP":
                if action_cache not in ["LEFT", "RIGHT", "UP", "DOWN"]:
                    ReleaseKey(direct_dic[action_cache.strip().split("_")[0]])
                    ReleaseKey(direct_dic[action_cache.strip().split("_")[1]])
                else:
                    ReleaseKey(direct_dic[action_cache])
                if not material:
                    PressKey(direct_dic["RIGHT"])
                    time.sleep(press_delay)
                    ReleaseKey(direct_dic["RIGHT"])
                    time.sleep(release_delay)
                    PressKey(direct_dic["RIGHT"])
                    time.sleep(press_delay)
                if material:
                    PressKey(direct_dic["RIGHT"])
                PressKey(direct_dic["UP"])
                # time.sleep(release_delay)
                action_cache = "RIGHT_UP"
                print("右上移动")
            else:
                print("右上移动")
        else:
            if not material:
                PressKey(direct_dic["RIGHT"])
                time.sleep(press_delay)
                ReleaseKey(direct_dic["RIGHT"])
                time.sleep(release_delay)
                PressKey(direct_dic["RIGHT"])
                time.sleep(press_delay)
            if material:
                PressKey(direct_dic["RIGHT"])
            PressKey(direct_dic["UP"])
            # time.sleep(press_delay)
            action_cache = "RIGHT_UP"
            print("右上移动")
        return action_cache

    elif direct == "RIGHT_DOWN":
        if action_cache != None:
            if action_cache != "RIGHT_DOWN":
                if action_cache not in ["LEFT", "RIGHT", "UP", "DOWN"]:
                    ReleaseKey(direct_dic[action_cache.strip().split("_")[0]])
                    ReleaseKey(direct_dic[action_cache.strip().split("_")[1]])
                else:
                    ReleaseKey(direct_dic[action_cache])
                if not material:
                    PressKey(direct_dic["RIGHT"])
                    time.sleep(press_delay)
                    ReleaseKey(direct_dic["RIGHT"])
                    time.sleep(release_delay)
                    PressKey(direct_dic["RIGHT"])
                    time.sleep(press_delay)
                if material:
                    PressKey(direct_dic["RIGHT"])
                PressKey(direct_dic["DOWN"])
                # time.sleep(press_delay)
                action_cache = "RIGHT_DOWN"
                print("右上移动")
            else:
                print("右上移动")
        else:
            if not material:
                PressKey(direct_dic["RIGHT"])
                time.sleep(press_delay)
                ReleaseKey(direct_dic["RIGHT"])
                time.sleep(release_delay)
                PressKey(direct_dic["RIGHT"])
                time.sleep(press_delay)
            if material:
                PressKey(direct_dic["RIGHT"])
            PressKey(direct_dic["DOWN"])
            # time.sleep(press_delay)
            action_cache = "RIGHT_DOWN"
            print("右上移动")
        return action_cache

    elif direct == "LEFT_UP":
        if action_cache != None:
            if action_cache != "LEFT_UP":
                if action_cache not in ["LEFT", "RIGHT", "UP", "DOWN"]:
                    ReleaseKey(direct_dic[action_cache.strip().split("_")[0]])
                    ReleaseKey(direct_dic[action_cache.strip().split("_")[1]])
                else:
                    ReleaseKey(direct_dic[action_cache])
                if not material:
                    PressKey(direct_dic["LEFT"])
                    time.sleep(press_delay)
                    ReleaseKey(direct_dic["LEFT"])
                    time.sleep(release_delay)
                    PressKey(direct_dic["LEFT"])
                    time.sleep(press_delay)
                if material:
                    PressKey(direct_dic["LEFT"])
                PressKey(direct_dic["UP"])
                # time.sleep(press_delay)
                action_cache = "LEFT_UP"
                print("左上移动")
            else:
                print("左上移动")
        else:
            if not material:
                PressKey(direct_dic["LEFT"])
                time.sleep(press_delay)
                ReleaseKey(direct_dic["LEFT"])
                time.sleep(release_delay)
                PressKey(direct_dic["LEFT"])
                time.sleep(press_delay)
            if material:
                PressKey(direct_dic["LEFT"])
            PressKey(direct_dic["UP"])
            # time.sleep(press_delay)
            action_cache = "LEFT_UP"
            print("左上移动")
        return action_cache

    elif direct == "LEFT_DOWN":
        if action_cache != None:
            if action_cache != "LEFT_DOWN":
                if action_cache not in ["LEFT", "RIGHT", "UP", "DOWN"]:
                    ReleaseKey(direct_dic[action_cache.strip().split("_")[0]])
                    ReleaseKey(direct_dic[action_cache.strip().split("_")[1]])
                else:
                    ReleaseKey(direct_dic[action_cache])
                if not material:
                    PressKey(direct_dic["LEFT"])
                    time.sleep(press_delay)
                    ReleaseKey(direct_dic["LEFT"])
                    time.sleep(release_delay)
                    PressKey(direct_dic["LEFT"])
                    time.sleep(press_delay)
                if material:
                    PressKey(direct_dic["LEFT"])
                PressKey(direct_dic["DOWN"])
                # time.sleep(press_delay)
                action_cache = "LEFT_DOWN"
                print("左下移动")
            else:
                print("左下移动")
        else:
            if not material:
                PressKey(direct_dic["LEFT"])
                time.sleep(press_delay)
                ReleaseKey(direct_dic["LEFT"])
                time.sleep(release_delay)
                PressKey(direct_dic["LEFT"])
                time.sleep(press_delay)
            if material:
                PressKey(direct_dic["LEFT"])
            PressKey(direct_dic["DOWN"])
            # time.sleep(press_delay)
            action_cache = "LEFT_DOWN"
            print("左下移动")
        return action_cache


if __name__ == "__main__":
    action_cache = None
    t1 = time.time()
    # while True:
        # if  int(time.time() - t1) % 2 == 0:
        #     action_cache = move("LEFT_DOWN", material=False, action_cache=action_cache, press_delay=0.1, release_delay=0.1)
        # else:
    action_cache = move("RIGHT_UP", material=True, action_cache=action_cache, press_delay=0.1, release_delay=0.1)