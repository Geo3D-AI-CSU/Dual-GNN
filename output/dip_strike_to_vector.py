import numpy as np
import math


def dip_strike_to_extension_vector(dip, strike, magnitude=1.0):
    """
    将倾角(dip)和走向(strike)转换为断层的延伸方向向量，并设置指定的模长

    参数:
    dip: 倾角，以度为单位，表示断层面与水平面的夹角，范围从0到90度
    strike: 走向，以度为单位，断层面与水平面交线相对于北方的角度，范围从0到360度
    magnitude: 返回向量的模长，默认为1.0

    返回:
    代表断层延伸方向的向量 [x, y, z]，长度为指定的模长
    """
    # 将角度转换为弧度
    dip_rad = math.radians(dip)
    strike_rad = math.radians(strike)

    # 计算倾向(dip direction)，它是走向顺时针旋转90度
    dip_direction_rad = math.radians((strike + 90) % 360)

    # 延伸方向向量与倾向在同一平面内，但与水平面成倾角
    # 北向为y轴正方向，东向为x轴正方向，垂直向下为z轴正方向
    x = np.sin(dip_direction_rad)  # 水平方向分量 x (东西方向)
    y = np.cos(dip_direction_rad)  # 水平方向分量 y (南北方向)
    z = np.tan(dip_rad)  # 垂直方向分量 (上下方向)，倾角越大，垂直分量越大

    # 计算向量长度用于归一化
    vector_length = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    # 返回按指定模长缩放的向量
    return [x / vector_length * magnitude,
            y / vector_length * magnitude,
            z / vector_length * magnitude]


# 示例使用
if __name__ == "__main__":
    # 示例：倾角85度，走向220度
    dip = 85
    strike = 220

    # 默认模长为1的单位向量
    vector1 = dip_strike_to_extension_vector(dip, strike)
    print(f"倾角 {dip}°, 走向 {strike}° 的单位延伸方向向量为: {vector1}")

    # 指定模长为2.5的向量
    vector2 = dip_strike_to_extension_vector(dip, strike, 400)
    print(f"倾角 {dip}°, 走向 {strike}° 的延伸方向向量(模长2.5)为: {vector2}")

    # 计算垂直分量与水平分量的比例，以检查垂直程度
    horizontal_mag = np.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
    vertical_mag = abs(vector1[2])
    print(f"垂直分量与水平分量的比例: {vertical_mag / horizontal_mag:.4f}")
    print(f"水平分量: {horizontal_mag:.4f}, 垂直分量: {vertical_mag:.4f}")