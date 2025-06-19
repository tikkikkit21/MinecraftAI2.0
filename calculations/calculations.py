import math
import pandas as pd
from typing import Tuple

# calculates length between 2 points
def calc_length(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

# calculates acute angle (in degrees) between 3 points
def calc_angle(p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> float:
    v1 = (p1[0] - p2[0], p1[1] - p2[1])
    v2 = (p3[0] - p2[0], p3[1] - p2[1])
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    len1 = math.sqrt(v1[0]**2 + v1[1]**2)
    len2 = math.sqrt(v2[0]**2 + v2[1]**2)

    if len1 == 0 or len2 == 0:
        return 0.0

    cos_angle = max(min(dot / (len1 * len2), 1.0), -1.0)
    angle_rad = math.acos(cos_angle)
    angle_deg = math.degrees(angle_rad)
    if angle_deg > 180:
        angle_deg = 360 - angle_deg
    return angle_deg

# calculates line orientation
def calc_line_orientation(base: Tuple[float, float], tip: Tuple[float, float]) -> float:
    dx = tip[0] - base[0]
    dy = tip[1] - base[1]
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad) % 360
    return angle_deg

# calculates angle orientation (think of it as an arrow like < points right)
def calc_vertex_angle(p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> float:
    v1 = (p2[0] - p1[0], p2[1] - p1[1])
    v2 = (p2[0] - p3[0], p2[1] - p3[1])

    def normalize(v):
        length = math.hypot(v[0], v[1])
        return (v[0]/length, v[1]/length) if length != 0 else (0.0, 0.0)

    n1 = normalize(v1)
    n2 = normalize(v2)

    bisector = (n1[0] + n2[0], n1[1] + n2[1])
    if bisector == (0.0, 0.0):
        bisector = (-n1[0], -n1[1])

    angle_rad = math.atan2(bisector[1], bisector[0])
    angle_deg = math.degrees(angle_rad)
    signed_angle = -(angle_deg - 90)

    if signed_angle > 180:
        signed_angle -= 360
    elif signed_angle < -180:
        signed_angle += 360

    return signed_angle

# helper function for naming x/y labels
def get_xy(row: pd.Series, label: str) -> Tuple[float, float]:
    return (row[f'{label}_x'], row[f'{label}_y'])
