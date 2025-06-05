import numpy as np
import math


def dip_strike_to_vector(dip, strike, magnitude=1.0):
    """
    将倾角(dip)和倾向(strike)转换为矢量，并乘上指定的模长

    参数:
    dip: 倾角，以度为单位，范围从0到90度
    strike: 倾向，以度为单位，范围从0到360度
    magnitude: 矢量的模长，默认为1.0

    返回:
    矢量 [x, y, z]，长度为指定的模长
    """
    # 将角度转换为弧度
    dip_rad = math.radians(dip)
    strike_rad = math.radians(strike)

    # 计算矢量分量
    # 注意：z轴向下为正方向
    x = np.sin(dip_rad) * np.sin(strike_rad)
    y = np.sin(dip_rad) * np.cos(strike_rad)
    z = np.cos(dip_rad)

    # 计算矢量长度用于归一化
    vector_length = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    # 计算带有指定模长的矢量
    vector = [x / vector_length * magnitude,
              y / vector_length * magnitude,
              z / vector_length * magnitude]

    return vector


# 示例使用
if __name__ == "__main__":
    # 示例：倾角85度，倾向220度
    dip = 85
    strike = 220

    # 默认模长为1（相当于单位矢量）
    vector1 = dip_strike_to_vector(dip, strike)
    print(f"倾角 {dip}°, 倾向 {strike}° 的单位矢量为: {vector1}")

    # 指定模长为2.5的矢量
    vector2 = dip_strike_to_vector(dip, strike, 400)
    print(f"倾角 {dip}°, 倾向 {strike}° 的矢量(模长2.5)为: {vector2}")