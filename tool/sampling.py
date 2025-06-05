import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString
from shapely.ops import linemerge, polygonize, unary_union

def sample_rock_unit_point_from_plate_figure(dem_csv_path, shapefile_info_list, output_csv_path, process_lines=False):
    # 读取所有矢量区域，并创建一个合并的 GeoDataFrame
    polygons = []
    for info in shapefile_info_list:
        shapefile_path = info['shapefile_path']
        attribute_value = info['attribute_value']

        # 读取矢量区域
        shp_gdf = gpd.read_file(shapefile_path)

        # 判断要素类型，如果是线并且需要处理，则转换为多边形
        geom_type = shp_gdf.geom_type.unique()
        if 'LineString' in geom_type or 'MultiLineString' in geom_type:
            if process_lines:
                print(f"正在将线要素 {shapefile_path} 转换为多边形...")
                # 将线要素转换为多边形
                shp_gdf = lines_to_polygons(shp_gdf)
            else:
                print(f"警告：{shapefile_path} 包含线要素，且未选择转换为多边形，可能无法正确进行空间查询。")
        elif 'Polygon' in geom_type or 'MultiPolygon' in geom_type:
            pass  # 如果是多边形，直接使用
        else:
            print(f"警告：{shapefile_path} 包含未知的几何类型：{geom_type}，将跳过此文件。")
            continue

        shp_gdf['attribute'] = attribute_value  # 添加属性列
        polygons.append(shp_gdf)

    if not polygons:
        print("未找到有效的矢量区域，程序结束。")
        return

    # 合并所有矢量区域
    polygons_gdf = gpd.GeoDataFrame(pd.concat(polygons, ignore_index=True))
    polygons_gdf = polygons_gdf[['geometry', 'attribute']]
    # 建立空间索引
    polygons_gdf.sindex

    # 读取整个 DEM 点数据，只读取需要的列
    print("正在读取 DEM 数据...")
    dem_df = pd.read_csv(dem_csv_path, usecols=['X', 'Y', 'Z'])
    print(f"DEM 数据已读取完毕，共 {len(dem_df)} 条记录。")

    # 修改列名以匹配代码中的命名
    dem_df.rename(columns={'X': 'x', 'Y': 'y', 'Z': 'z'}, inplace=True)
    dem_df = dem_df[['x', 'y', 'z']]  # 只保留 x, y, z 列

    # 将 DEM 数据转换为 GeoDataFrame
    dem_gdf = gpd.GeoDataFrame(
        dem_df,
        geometry=gpd.points_from_xy(dem_df['x'], dem_df['y'])
    )
    # 设置坐标系（如果已知）
    # dem_gdf.crs = 'EPSG:XXXX'
    # polygons_gdf.crs = 'EPSG:XXXX'

    # 初始化属性列
    dem_gdf['attribute'] = None

    # 使用空间连接，找到 DEM 点与矢量区域的关系
    print("正在进行空间连接，这可能需要一些时间...")
    dem_with_attr = gpd.sjoin(dem_gdf, polygons_gdf, how='left', predicate='intersects')
    # 如果存在多个匹配（点位于多个区域），可以选择保留第一个匹配
    dem_with_attr = dem_with_attr.drop_duplicates(subset=['x', 'y'])
    # 删除多余的列
    dem_with_attr = dem_with_attr.drop(columns=['index_right', 'geometry'])

    # 只保留需要的列
    dem_with_attr = dem_with_attr[['x', 'y', 'z', 'attribute_right']]

    # 保存结果
    print("正在保存结果...")
    dem_with_attr.to_csv(output_csv_path, index=False)
    print(f"已生成 CSV 文件：{output_csv_path}")

def lines_to_polygons(lines_gdf):
    """
    将线要素 GeoDataFrame 转换为多边形 GeoDataFrame。
    """
    # 合并所有线要素
    lines = [geom for geom in lines_gdf.geometry]
    merged_lines = linemerge(lines)

    # 处理未闭合的线，将起点和终点连接
    if isinstance(merged_lines, LineString):
        if not merged_lines.is_ring:
            coords = list(merged_lines.coords)
            coords.append(coords[0])  # 将起点添加到末尾
            merged_lines = LineString(coords)
        merged_lines = [merged_lines]
    elif isinstance(merged_lines, list):
        merged_lines = [LineString(line.coords) if not line.is_ring else line for line in merged_lines]

    # 构建多边形
    borders = unary_union(merged_lines)
    polygons = list(polygonize(borders))

    # 创建 GeoDataFrame 保存多边形
    polygons_gdf = gpd.GeoDataFrame(geometry=polygons)
    polygons_gdf.crs = lines_gdf.crs  # 保持坐标系一致

    return polygons_gdf


from osgeo import ogr, osr

def dxf_lines_to_shapefile(dxf_path, shapefile_path):
    """
    将 DXF 文件中的线要素（包括三维线）转换为 Shapefile 格式。

    参数：
    - dxf_path: 输入 DXF 文件的路径。
    - shapefile_path: 输出 Shapefile 文件的路径（包括文件名和 .shp 扩展名）。

    返回：
    无，结果将保存为指定的 Shapefile 文件。
    """
    # 注册所有驱动
    ogr.RegisterAll()

    # 打开 DXF 文件
    dxf_datasource = ogr.Open(dxf_path)
    if dxf_datasource is None:
        print(f"无法打开 DXF 文件：{dxf_path}")
        return

    # 获取 Shapefile 驱动
    shapefile_driver = ogr.GetDriverByName("ESRI Shapefile")
    if shapefile_driver is None:
        print("无法获取 ESRI Shapefile 驱动。")
        return

    # 创建输出数据源
    shapefile_datasource = shapefile_driver.CreateDataSource(shapefile_path)
    if shapefile_datasource is None:
        print(f"无法创建 Shapefile 数据源：{shapefile_path}")
        return

    # 创建 Shapefile 图层，几何类型为三维 LineString
    out_layer = shapefile_datasource.CreateLayer("lines", geom_type=ogr.wkbLineString25D)
    if out_layer is None:
        print("无法创建 Shapefile 图层。")
        return

    # 添加必要的字段（可选）
    # 示例：添加一个名为 'LayerName' 的字符串字段，长度为50
    field_defn = ogr.FieldDefn("LayerName", ogr.OFTString)
    field_defn.SetWidth(50)
    out_layer.CreateField(field_defn)

    # 获取 DXF 文件中的图层数量
    layer_count = dxf_datasource.GetLayerCount()

    # 遍历每个图层
    for i in range(layer_count):
        layer = dxf_datasource.GetLayer(i)
        layer_defn = layer.GetLayerDefn()
        layer_name_dxf = layer_defn.GetName()

        print(f"正在处理 DXF 图层：{layer_name_dxf}")

        # 遍历要素
        for feature in layer:
            geom = feature.GetGeometryRef()
            if geom is None:
                continue

            geom_type = geom.GetGeometryType()

            # 处理 LineString 和 LineString25D 类型的要素
            if geom_type in [ogr.wkbLineString, ogr.wkbLineString25D]:
                # 创建新的要素
                out_feature = ogr.Feature(out_layer.GetLayerDefn())
                out_feature.SetGeometry(geom.Clone())

                # 设置属性字段（可选）
                out_feature.SetField("LayerName", layer_name_dxf)

                out_layer.CreateFeature(out_feature)
                out_feature = None  # 清除引用

            elif geom_type in [ogr.wkbMultiLineString, ogr.wkbMultiLineString25D]:
                # 处理 MultiLineString，将其分解为单独的 LineString
                for j in range(geom.GetGeometryCount()):
                    single_geom = geom.GetGeometryRef(j)

                    # 创建新的要素
                    out_feature = ogr.Feature(out_layer.GetLayerDefn())
                    out_feature.SetGeometry(single_geom.Clone())

                    # 设置属性字段（可选）
                    out_feature.SetField("LayerName", layer_name_dxf)

                    out_layer.CreateFeature(out_feature)
                    out_feature = None  # 清除引用

            else:
                # 跳过其他类型的要素
                continue

            feature = None  # 清除引用

        layer = None  # 清除引用

    # 清除数据源引用
    dxf_datasource = None
    shapefile_datasource = None

    print(f"转换完成，Shapefile 已保存到：{shapefile_path}")
# dem_csv = 'data/gz/DEM.csv'
# shapefile_info_list = [
#     {
#         'shapefile_path': 'data/gz/rock_unit/plate_figure/shapefile/PTM1_ROCK_UNIT1.SHP',
#         'attribute_value': '1'
#     },
#     {
#         'shapefile_path': 'data/gz/rock_unit/plate_figure/shapefile/PTM2_ROCK_UNIT2.SHP',
#         'attribute_value': '2'
#     },
#     {
#         'shapefile_path': 'data/gz/rock_unit/plate_figure/shapefile/PTM3_ROCK_UNIT3.SHP',
#         'attribute_value': '3'
#     },
#     {
#         'shapefile_path': 'data/gz/rock_unit/plate_figure/shapefile/PTM4_ROCK_UNIT4.SHP',
#         'attribute_value': '4'
#     },
#     {
#         'shapefile_path': 'data/gz/rock_unit/plate_figure/shapefile/PTM5_ROCK_UNIT5.SHP',
#         'attribute_value': '5'
#     },
#     {
#         'shapefile_path': 'data/gz/rock_unit/plate_figure/shapefile/PTT_ROCK_UNIT6.SHP',
#         'attribute_value': '6'
#     },
#     {
#         'shapefile_path': 'data/gz/rock_unit/plate_figure/shapefile/PTX_ROCK_UNIT0.SHP',
#         'attribute_value': '0'
#     },
#     {
#         'shapefile_path': 'data/gz/rock_unit/plate_figure/shapefile/K_UNIT7.SHP',
#         'attribute_value': '7'
#     },
# ]
# output_csv_path = 'data/gz/plata_figure_rock_unit.csv'
# sample_rock_unit_point_from_plate_figure(dem_csv, shapefile_info_list, output_csv_path, process_lines=True)


import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString
import numpy as np
from scipy.spatial import cKDTree

def sample_line_points_on_dem(dem_csv_path, shapefile_info_list, output_csv_path, interval):
    """
    将多个二维线数据投影到DEM上，按照指定间距在线上生成离散采样点，获取对应的Z值，并保存为CSV格式。

    参数：
    - dem_csv_path: DEM离散点数据的CSV文件路径，包含X, Y, Z列。
    - shapefile_info_list: 包含每个线文件信息的列表，每个元素是一个字典：
        {
            'shapefile_path': 线Shapefile文件路径,
            'attribute_value': 属性值
        }
    - output_csv_path: 输出CSV文件的路径。
    - interval: 在线上生成采样点的间距（单位与坐标系一致）。

    返回：
    无，结果将保存为指定的CSV文件。
    """
    # 读取DEM数据
    dem_df = pd.read_csv(dem_csv_path)
    dem_points = dem_df[['X', 'Y']].values
    dem_z = dem_df['Z'].values

    # 建立DEM点的KDTree用于快速查询
    dem_tree = cKDTree(dem_points)

    # 存储结果的列表
    results = []

    # 遍历每个线文件
    for info in shapefile_info_list:
        shapefile_path = info['shapefile_path']
        attribute_value = info['attribute_value']

        # 读取线数据
        line_gdf = gpd.read_file(shapefile_path)

        # 添加属性列
        line_gdf['attribute'] = attribute_value

        # 遍历每条线
        for idx, row in line_gdf.iterrows():
            geometry = row.geometry
            if geometry is None:
                continue

            # 获取该线的属性值
            attribute_value = row['attribute']

            # 如果是MultiLineString，拆分为多个LineString
            if isinstance(geometry, MultiLineString):
                lines = list(geometry)
            elif isinstance(geometry, LineString):
                lines = [geometry]
            else:
                # 非线类型，跳过
                continue

            # 对每个LineString生成采样点
            for line in lines:
                # 计算线的长度
                length = line.length
                if length == 0:
                    continue  # 跳过长度为0的线

                # 生成采样点的距离序列
                num_samples = max(int(length // interval) + 1, 2)
                distances = np.linspace(0, length, num_samples)
                # 生成采样点
                for distance in distances:
                    point = line.interpolate(distance)
                    x, y = point.x, point.y
                    # 查询最近的DEM点
                    dist, idx_dem = dem_tree.query([x, y], k=1)
                    z = dem_z[idx_dem]
                    # 保存结果
                    results.append({'X': x, 'Y': y, 'Z': z, 'attribute': attribute_value})

    # 将结果保存为DataFrame
    result_df = pd.DataFrame(results)

    # 保存为CSV文件
    result_df.to_csv(output_csv_path, index=False)
    print(f"已生成CSV文件：{output_csv_path}")

# shapefile_info_list = [
#     {
#         'shapefile_path': 'data/gz/rock_unit/plate_figure/shapefile/level/LEVEL_H6.SHP',
#         'attribute_value': '6'
#     },
#     {
#         'shapefile_path': 'data/gz/rock_unit/plate_figure/shapefile/level/LEVEL_H5.SHP',
#         'attribute_value': '5'
#     },
#     {
#         'shapefile_path': 'data/gz/rock_unit/plate_figure/shapefile/level/LEVEL_H4.SHP',
#         'attribute_value': '4'
#     },
#     {
#         'shapefile_path': 'data/gz/rock_unit/plate_figure/shapefile/level/LEVEL_H3.SHP',
#         'attribute_value': '3'
#     },
#     {
#         'shapefile_path': 'data/gz/rock_unit/plate_figure/shapefile/level/LEVEL_H2.SHP',
#         'attribute_value': '2'
#     },
#     {
#         'shapefile_path': 'data/gz/rock_unit/plate_figure/shapefile/level/LEVEL_F38.SHP',
#         'attribute_value': '38'
#     },
#     {
#         'shapefile_path': 'data/gz/rock_unit/plate_figure/shapefile/level/LEVEL_F37_1.SHP',
#         'attribute_value': '37.1'
#     },
#     {
#         'shapefile_path': 'data/gz/rock_unit/plate_figure/shapefile/level/LEVEL_F37.SHP',
#         'attribute_value': '37'
#     },
#     # 可以继续添加更多的线文件信息
# ]
# output_csv_path = 'data/gz/plata_figure_level.csv'
# interval = 10  # 采样间距为10单位
#
# sample_line_points_on_dem(dem_csv, shapefile_info_list, output_csv_path, interval)




