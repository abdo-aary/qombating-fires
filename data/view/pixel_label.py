import geopandas as gpd
from shapely.geometry import Point

# Charger le shapefile contenant les océans
url = 'https://naciscdn.org/naturalearth/110m/physical/ne_110m_ocean.zip'
ocean_gdf = gpd.read_file(url)


def is_point_in_ocean(point, ocean_gdf):
    return ocean_gdf.contains(point).any()

    return results


def is_cell_in_ocean(lat, lon, ocean_gdf, delta=0.125):
    """
    Vérifie si une cellule est entièrement dans l'océan en testant son centre et ses 4 coins.
    """
    points = [
        Point(lon, lat),
        Point(lon + delta, lat + delta),
        Point(lon - delta, lat - delta),
        Point(lon - delta, lat + delta),
        Point(lon + delta, lat - delta)
    ]

    for point in points:
        if is_point_in_ocean(point, ocean_gdf) == False:
            return False
    return True


def is_cell_in_quebec(lat, lon, quebec, delta=0.125):
    """
    Vérifie si une cellule est entièrement dans l'océan en testant son centre et ses 4 coins.
    """
    points = [
        Point(lon, lat),
        Point(lon + delta, lat + delta),
        Point(lon - delta, lat - delta),
        Point(lon - delta, lat + delta),
        Point(lon + delta, lat - delta)
    ]

    for point in points:
        if quebec.geometry.contains(point).any():
            return True
    return False