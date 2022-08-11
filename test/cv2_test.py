import cv2
from ..grabscreen import grab_screen

img = grab_screen((0, 0, 1280, 800))
img0 = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
dnf = cv2.imread(img0)
cv2.imshow('rgb', dnf)
# BGR = cv2.cvtColor(dnf,cv2.COLOR_RGB2BGR)
# cv2.imshow('BGR',BGR)