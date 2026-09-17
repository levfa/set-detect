import cv2
import numpy as np

# Backend-dependent cv2.waitKey codes for arrow keys.
ARROW_CODES: dict[int, str] = {
    81: "left",
    82: "up",
    83: "right",
    84: "down",
    65361: "left",
    65362: "up",
    65363: "right",
    65364: "down",
    16777234: "left",
    16777235: "up",
    16777236: "right",
    16777237: "down",
    2424832: "left",
    2490368: "up",
    2555904: "right",
    2621440: "down",
}

ESC_KEY = 27

CORNER_COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]

COUNT_LETTERS = {0: "1", 1: "2", 2: "3", 3: "?"}
COLOR_LETTERS = {0: "r", 1: "g", 2: "p", 3: "?"}
SHAPE_LETTERS = {0: "d", 1: "e", 2: "q", 3: "?"}
FILL_LETTERS = {0: "o", 1: "s", 2: "t", 3: "?"}


def draw_label(img: np.ndarray, text: str, org: tuple[int, int]) -> None:
    # draw a thick black outline under the white fill so the text stays readable
    # over any background
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
