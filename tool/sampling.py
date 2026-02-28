import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString
from shapely.ops import linemerge, polygonize, unary_union
from shapely.geometry import LineString, MultiLineString
import numpy as np
from scipy.spatial import cKDTree
from osgeo import ogr, osr

def sample_rock_unit_point_from_plate_figure(dem_csv_path, shapefile_info_list, output_csv_path, process_lines=False):
    # Read all vector areas and create a merged GeoDataFrame
    polygons = []
    for info in shapefile_info_list:
        shapefile_path = info['shapefile_path']
        attribute_value = info['attribute_value']

        # Read vector area
        shp_gdf = gpd.read_file(shapefile_path)

        # Determine the feature type; if it is a line and requires processing, convert it to a polygon
        geom_type = shp_gdf.geom_type.unique()
        if 'LineString' in geom_type or 'MultiLineString' in geom_type:
            if process_lines:
                shp_gdf = lines_to_polygons(shp_gdf)
            else:
                print(f"{shapefile_path}")
        elif 'Polygon' in geom_type or 'MultiPolygon' in geom_type:
            pass  
        else:
            print(f"{shapefile_path} ")
            continue

        shp_gdf['attribute'] = attribute_value 
        polygons.append(shp_gdf)

    if not polygons:
        return

    # Merge all vector areas
    polygons_gdf = gpd.GeoDataFrame(pd.concat(polygons, ignore_index=True))
    polygons_gdf = polygons_gdf[['geometry', 'attribute']]
    # Establish spatial indexes
    polygons_gdf.sindex

    # Read the entire DEM point data, retrieving only the required columns
    dem_df = pd.read_csv(dem_csv_path, usecols=['X', 'Y', 'Z'])

    # Modify column names to match the naming conventions in the code
    dem_df.rename(columns={'X': 'x', 'Y': 'y', 'Z': 'z'}, inplace=True)
    dem_df = dem_df[['x', 'y', 'z']]  

    # Convert DEM data to GeoDataFrame
    dem_gdf = gpd.GeoDataFrame(
        dem_df,
        geometry=gpd.points_from_xy(dem_df['x'], dem_df['y'])
    )

    dem_gdf['attribute'] = None

    # Utilise spatial join to determine the relationship between DEM points and vector areas
    dem_with_attr = gpd.sjoin(dem_gdf, polygons_gdf, how='left', predicate='intersects')
    # If multiple matches exist, the first match may be retained
    dem_with_attr = dem_with_attr.drop_duplicates(subset=['x', 'y'])
    # Remove redundant columns
    dem_with_attr = dem_with_attr.drop(columns=['index_right', 'geometry'])

    # Retain only the required columns
    dem_with_attr = dem_with_attr[['x', 'y', 'z', 'attribute_right']]
    dem_with_attr.to_csv(output_csv_path, index=False)

def lines_to_polygons(lines_gdf):
    """Convert line feature GeoDataFrame to polygon GeoDataFrame"""
    
    lines = [geom for geom in lines_gdf.geometry]
    merged_lines = linemerge(lines)

    # Process unclosed lines by connecting their start and end points
    if isinstance(merged_lines, LineString):
        if not merged_lines.is_ring:
            coords = list(merged_lines.coords)
            coords.append(coords[0]) 
            merged_lines = LineString(coords)
        merged_lines = [merged_lines]
    elif isinstance(merged_lines, list):
        merged_lines = [LineString(line.coords) if not line.is_ring else line for line in merged_lines]

    # Constructing polygons
    borders = unary_union(merged_lines)
    polygons = list(polygonize(borders))

    # Create a GeoDataFrame to store polygons
    polygons_gdf = gpd.GeoDataFrame(geometry=polygons)
    polygons_gdf.crs = lines_gdf.crs  

    return polygons_gdf


def dxf_lines_to_shapefile(dxf_path, shapefile_path):
    """Convert line features (including 3D lines) from DXF files to Shapefile format"""
    
    ogr.RegisterAll()
    dxf_datasource = ogr.Open(dxf_path)
    if dxf_datasource is None:
        return

    # Obtain Shapefile driver
    shapefile_driver = ogr.GetDriverByName("ESRI Shapefile")
    if shapefile_driver is None:
        return

    # Create an output data source
    shapefile_datasource = shapefile_driver.CreateDataSource(shapefile_path)
    if shapefile_datasource is None:
        return

    # Create a Shapefile layer with the geometry type set to 3D LineString
    out_layer = shapefile_datasource.CreateLayer("lines", geom_type=ogr.wkbLineString25D)
    if out_layer is None:
        return

    # Add the necessary fields
    field_defn = ogr.FieldDefn("LayerName", ogr.OFTString)
    field_defn.SetWidth(50)
    out_layer.CreateField(field_defn)

    # Retrieve the number of layers in a DXF file
    layer_count = dxf_datasource.GetLayerCount()

    # Iterate through each layer
    for i in range(layer_count):
        layer = dxf_datasource.GetLayer(i)
        layer_defn = layer.GetLayerDefn()
        layer_name_dxf = layer_defn.GetName()

        # Traverse elements
        for feature in layer:
            geom = feature.GetGeometryRef()
            if geom is None:
                continue

            geom_type = geom.GetGeometryType()

            # Processing features of the LineString and LineString25D types
            if geom_type in [ogr.wkbLineString, ogr.wkbLineString25D]:
                # Create new features
                out_feature = ogr.Feature(out_layer.GetLayerDefn())
                out_feature.SetGeometry(geom.Clone())

                # Set property fields
                out_feature.SetField("LayerName", layer_name_dxf)

                out_layer.CreateFeature(out_feature)
                out_feature = None  # Clear references

            elif geom_type in [ogr.wkbMultiLineString, ogr.wkbMultiLineString25D]:
                # Process MultiLineString, breaking it down into individual LineStrings
                for j in range(geom.GetGeometryCount()):
                    single_geom = geom.GetGeometryRef(j)

                    # Create new features
                    out_feature = ogr.Feature(out_layer.GetLayerDefn())
                    out_feature.SetGeometry(single_geom.Clone())

                    # Set property fields
                    out_feature.SetField("LayerName", layer_name_dxf)
                    out_layer.CreateFeature(out_feature)
                    out_feature = None  # Clear references

            else:
                # Skip other types of elements
                continue
            feature = None  
        layer = None  

    # Clear data source references
    dxf_datasource = None
    shapefile_datasource = None


def sample_line_points_on_dem(dem_csv_path, shapefile_info_list, output_csv_path, interval):
    # Reading DEM data
    dem_df = pd.read_csv(dem_csv_path)
    dem_points = dem_df[['X', 'Y']].values
    dem_z = dem_df['Z'].values

    # The KDTree constructed for DEM points facilitates rapid queries.
    dem_tree = cKDTree(dem_points)
    results = []

    # Iterate through each line file
    for info in shapefile_info_list:
        shapefile_path = info['shapefile_path']
        attribute_value = info['attribute_value']

        # Read line data
        line_gdf = gpd.read_file(shapefile_path)

        # Add attribute column
        line_gdf['attribute'] = attribute_value

        # Traverse each line
        for idx, row in line_gdf.iterrows():
            geometry = row.geometry
            if geometry is None:
                continue

            # Retrieve the property values for this line
            attribute_value = row['attribute']

            # If it is a MultiLineString, split it into multiple LineStrings
            if isinstance(geometry, MultiLineString):
                lines = list(geometry)
            elif isinstance(geometry, LineString):
                lines = [geometry]
            else:
                # Non-linear type, skip
                continue

            # Generate sampling points for each LineString
            for line in lines:
                # Calculate the length of the line
                length = line.length
                if length == 0:
                    continue 

                # Generate a distance sequence for sampling points
                num_samples = max(int(length // interval) + 1, 2)
                distances = np.linspace(0, length, num_samples)
                # Generate sampling points
                for distance in distances:
                    point = line.interpolate(distance)
                    x, y = point.x, point.y
                    # Query the nearest DEM point
                    dist, idx_dem = dem_tree.query([x, y], k=1)
                    z = dem_z[idx_dem]
                    # Save results
                    results.append({'X': x, 'Y': y, 'Z': z, 'attribute': attribute_value})

    # Save the results as a DataFrame
    result_df = pd.DataFrame(results)
    # Save as CSV file
    result_df.to_csv(output_csv_path, index=False)







