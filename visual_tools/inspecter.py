# import os
# from datetime import datetime
# import earthaccess
# from pystac_client import Client
# from shapely.geometry import box

# # 1. Define your geographic area (Bounding Box) and target year
# # Example: Agriculture zone in California [minx, miny, maxx, maxy]
# bbox = [16.18, 48.12, 16.59, 48.34]       
# target_year = 2024
# start_date = f"{target_year}-06-01T00:00:00Z"
# end_date = f"{target_year}-08-31T23:59:59Z"

# output_dir = "./satellite_data"
# os.makedirs(output_dir, exist_ok=True)

# # =========================================================================
# # STEP 1: Authenticate and Search NASA Earthdata for ECOSTRESS
# # =========================================================================
# print("Logging into NASA Earthdata...")
# # This will look for environment variables or prompt you inline the first time
# auth = earthaccess.login(strategy="interactive") 

# print("Searching for ECOSTRESS L2 LSTE data...")
# # Concept ID for ECOSTRESS Land Surface Temperature & Emissivity V002
# # ecostress_concept_id = "C2314352125-LPCLOUD" 

# eco_results = earthaccess.search_data(
#     short_name="ECO_L2T_LSTE",
#     version="002",
#     bounding_box=tuple(bbox),
#     temporal=(start_date[:10], end_date[:10])
# )


# # Extract unique dates (YYYY-MM-DD) where ECOSTRESS has coverage
# eco_dates = set()
# eco_granules_by_date = {}

# for granule in eco_results:
#     # Parse time metadata string
#     time_str = granule['umm']['TemporalExtent']['RangeDateTime']['BeginningDateTime']
#     date_key = time_str.split("T")[0] # Get 'YYYY-MM-DD'
#     eco_dates.add(date_key)
    
#     if date_key not in eco_granules_by_date:
#         eco_granules_by_date[date_key] = []
#     eco_granules_by_date[date_key].append(granule)

# print(f"Found ECOSTRESS scenes across {len(eco_dates)} unique days.")

# # =========================================================================
# # STEP 2: Query Sentinel-2 STAC for the same spatial/temporal bounds
# # =========================================================================
# print("Querying Sentinel-2 STAC archive...")
# # Change this line in inspecter.py:
# stac_client = Client.open("https://earth-search.aws.element84.com/v1/")
# roi_geometry = box(*bbox).__geo_interface__

# s2_search = stac_client.search(
#     collections=["sentinel-2-l2a"],
#     intersects=roi_geometry,
#     datetime=f"{start_date}/{end_date}",
#     query={"eo:cloud_cover": {"lt": 15}} # Filter out cloudy days
# )
# s2_items = s2_search.item_collection()

# # Organize Sentinel-2 scenes by their calendar date
# s2_dates = set()
# s2_items_by_date = {}

# for item in s2_items:
#     date_key = item.properties["datetime"].split("T")[0]
#     s2_dates.add(date_key)
#     if date_key not in s2_items_by_date:
#         s2_items_by_date[date_key] = []
#     s2_items_by_date[date_key].append(item)

# print(f"Found clean Sentinel-2 scenes across {len(s2_dates)} unique days.")

# # # =========================================================================
# # # STEP 3: Match the Days of the Year (DOY) and Download Pairs
# # # =========================================================================
# matching_dates = list(eco_dates.intersection(s2_dates))
# print(f"\nFound {len(matching_dates)} perfectly matching calendar days!")

# for date_id in matching_dates:
#     print(f"{date_id}\n")



# for date_str in sorted(list(matching_dates)):
#     doy = datetime.strptime(date_str, "%Y-%m-%d").timetuple().tm_yday
#     print(f"\n--- Processing Date: {date_str} (DOY: {doy}) ---")
    
#     day_dir = os.path.join(output_dir, date_str)
#     os.makedirs(day_dir, exist_ok=True)
    
#     # A. Download ECOSTRESS for this day
#     print(f" Downloading ECOSTRESS data layer...")
#     eco_granules = eco_granules_by_date[date_str]
#     # Downloads the HDF5/GeoTIFF files directly into the day's folder
#     earthaccess.download(eco_granules, local_path=day_dir)
    
#     # B. Download Sentinel-2 Assets for this day (B4 and B8 for NDVI)
#     print(f" Downloading Sentinel-2 optical bands (Red/NIR)...")
#     s2_item = s2_items_by_date[date_str][0] # Grab the first available scene asset
    
#     for band in ["red", "nir"]:
#         asset_url = s2_item.assets[band].href
#         filename = f"sentinel2_{band}.tif"
#         dest_path = os.path.join(day_dir, filename)
        
#         # Stream the asset locally using earthaccess or direct requests
#         # (Alternatively, keep cloud-optimized urls for direct stackstac processing)
#         print(f"   Saving {band} band asset to {dest_path}")
#         # Note: You can use earthaccess.download_urls if assets sit on AWS or standard streaming


import os
from datetime import datetime
import earthaccess
from pystac_client import Client
from shapely.geometry import box
import cdsapi

# 1. Define your geographic area (Bounding Box) and target year
# Example: Agriculture zone in California [minx, miny, maxx, maxy]
bbox = [16.18, 48.12, 16.59, 48.34]  
target_year = 2024
start_date = f"{target_year}-06-01T00:00:00Z"
end_date = f"{target_year}-08-31T23:59:59Z"

output_dir = "./satellite_data_downloaded"
os.makedirs(output_dir, exist_ok=True)

# Initialize Copernicus Climate Data Store client
cds_client = cdsapi.Client()

# =========================================================================
# STEP 1: Authenticate and Search NASA Earthdata for ECOSTRESS
# =========================================================================
print("Logging into NASA Earthdata...")
auth = earthaccess.login(strategy="interactive") 

print("Searching for ECOSTRESS L2 LSTE data...")
# ecostress_concept_id = "C2314352125-LPCLOUD" 

# FIX: Changed 'bbox' to 'bounding_box' to meet earthaccess structural validation
eco_results = earthaccess.search_data(
    short_name="ECO_L2T_LSTE",
    version="002",
    bounding_box=tuple(bbox),
    temporal=(start_date[:10], end_date[:10])
)


eco_dates = set()
eco_granules_by_date = {}

for granule in eco_results:
    time_str = granule['umm']['TemporalExtent']['RangeDateTime']['BeginningDateTime']
    date_key = time_str.split("T")[0] 
    eco_dates.add(date_key)
    
    if date_key not in eco_granules_by_date:
        eco_granules_by_date[date_key] = []
    eco_granules_by_date[date_key].append(granule)

print(f"Found ECOSTRESS scenes across {len(eco_dates)} unique days.")

# =========================================================================
# STEP 2: Query Sentinel-2 STAC for the same spatial/temporal bounds
# =========================================================================
print("Querying Sentinel-2 STAC archive...")
# FIX: Corrected root JSON API endpoint for Element 84 Earth Search
stac_client = Client.open("https://earth-search.aws.element84.com/v1/")
# FIX: Replaced invalid __geometry__ attribute call with standard __geo_interface__ dictionary map
roi_geometry = box(*bbox).__geo_interface__

s2_search = stac_client.search(
    collections=["sentinel-2-l2a"],
    intersects=roi_geometry,
    datetime=f"{start_date}/{end_date}",
    query={"eo:cloud_cover": {"lt": 15}} 
)
s2_items = s2_search.item_collection()

s2_dates = set()
s2_items_by_date = {}

for item in s2_items:
    date_key = item.properties["datetime"].split("T")[0]
    s2_dates.add(date_key)
    if date_key not in s2_items_by_date:
        s2_items_by_date[date_key] = []
    s2_items_by_date[date_key].append(item)

print(f"Found clean Sentinel-2 scenes across {len(s2_dates)} unique days.")

# =========================================================================
# STEP 3: Match the Days of the Year (DOY) and Download Everything
# =========================================================================
matching_dates = eco_dates.intersection(s2_dates)
print(f"\nFound {len(matching_dates)} perfectly matching calendar days!")

for date_id in matching_dates:
    print(f"{date_id}\n")

for date_str in sorted(list(matching_dates)):
    dt_obj = datetime.strptime(date_str, "%Y-%m-%d")
    doy = dt_obj.timetuple().tm_yday
    print(f"\n--- Processing Date: {date_str} (DOY: {doy}) ---")
    
    day_dir = os.path.join(output_dir, date_str)  
    os.makedirs(day_dir, exist_ok=True)
    
    # A. Download ECOSTRESS for this day
    print(f" Downloading ECOSTRESS data layer...")
    eco_granules = eco_granules_by_date[date_str]
    earthaccess.download(eco_granules, local_path=day_dir)
    
    # B. Download Sentinel-2 Assets for this day (B4 and B8 for NDVI)
    print(f" Downloading Sentinel-2 optical bands (Red/NIR)...")
    s2_item = s2_items_by_date[date_str][0] 
    
    for band in ["red", "nir"]:
        asset_url = s2_item.assets[band].href
        filename = f"sentinel2_{band}.tif"
        dest_path = os.path.join(day_dir, filename)
        print(f"   Saving {band} band asset to {dest_path}")
        # Add actual download logic here if not using streamed COGs
        
    # C. Download Coincident Daily ERA5 Reanalysis Covariates
    print(f" Requesting ERA5 Daily Reanalysis Parameters...")
    era5_filename = os.path.join(day_dir, f"era5_daily_{date_str}.nc")
    
    # Convert bounding box coordinate layout to ERA5 expectations [North, West, South, East]
    # bbox format: [minx, miny, maxx, maxy] -> [West, South, East, North]
    era5_area = [bbox[3], bbox[0], bbox[1], bbox[2]] 
    
#    try:
#        cds_client.retrieve(
#            'reanalysis-era5-single-levels',
#            {
#                'product_type': 'reanalysis',
#                'format': 'netcdf',
#                'variable': [
#                    '2m_temperature', 
#                    'total_precipitation',
#                    'skin_temperature'
#                ],
#                'year': str(dt_obj.year),
#                'month': f"{dt_obj.month:02d}",
#                'day': f"{dt_obj.day:02d}",
#                'time': [f"{h:02d}:00" for h in range(24)], # Request all 24 hours to build daily values
#                'area': era5_area,
#            },
#            era5_filename
#        )
#        print(f"   Successfully saved ERA5 NetCDF to {era5_filename}")
#    except Exception as e:
#        print(f"   Error fetching ERA5 data for {date_str}: {e}")

print("\nData collection complete across matching observation timelines.")







# ############## Preprocessing ERA4 dataset  ######################
# import xarray as xr
# import rioxarray
# from rasterio.enums import Resampling

# def process_era5_to_match_sentinel(era5_path, sentinel_template_path, output_dir):
#     """
#     Opens the downloaded hourly ERA5 NetCDF file, calculates the daily mean,
#     reprojects/resamples it to match the precise 30m grid projection of the 
#     Sentinel-2 template image, and exports it as individual GeoTIFF files.
#     """
#     print(f"--- Extracting and Gridding ERA5 Data from: {os.path.basename(era5_path)} ---")
    
#     # 1. Open NetCDF file using xarray
#     # ERA5 single levels typically store coordinate data under 'latitude' and 'longitude'
#     ds = xr.open_dataset(era5_path)
    
#     # Set spatial dimensions and source Coordinate Reference System (CRS is EPSG:4326 for ERA5)
#     ds = ds.rio.set_spatial_dims(x_dim='longitude', y_dim='latitude')
#     ds.rio.write_crs("EPSG:4326", inplace=True)
    
#     # 2. Compute the daily average across the 24-hour time axis
#     daily_mean = ds.mean(dim='time')
    
#     # 3. Load the Sentinel-2 image metadata structure to act as the target grid layout
#     with rioxarray.open_rasterio(sentinel_template_path) as s2_img:
#         target_crs = s2_img.rio.crs
#         target_transform = s2_img.rio.transform
#         target_width = s2_img.rio.width
#         target_height = s2_img.rio.height
    
#     # 4. Loop through every variable inside the ERA5 file (e.g., t2m, tp, skt)
#     for var_name in daily_mean.data_vars:
#         data_array = daily_mean[var_name]
        
#         # Convert Temperature values from Kelvin to Celsius if necessary
#         if 'temperature' in data_array.attrs.get('long_name', '').lower() or 't2m' in var_name or 'skt' in var_name:
#             data_array = data_array - 273.15
#             data_array.attrs['units'] = 'degrees Celsius'
            
#         print(f"   Reprojecting variable '{var_name}' to match 30m Sentinel-2 grid...")
        
#         # Reproject and resample from the coarse (~25km) ERA5 resolution to the fine 30m grid layout
#         # Bilinear interpolation ensures smooth transitions over the regional atmospheric variables
#         resampled_array = data_array.rio.reproject(
#             dst_crs=target_crs,
#             shape=(target_height, target_width),
#             transform=target_transform,
#             resampling=Resampling.bilinear
#         )
        
#         # 5. Export clean single-band GeoTIFF file
#         output_tif_path = os.path.join(output_dir, f"era5_30m_{var_name}.tif")
#         resampled_array.rio.to_raster(output_tif_path)
#         print(f"   Successfully saved: {output_tif_path}")
        
#     ds.close()

# # =========================================================================
# # EXAMPLE USAGE INSIDE YOUR EXISTING DATE LOOP:
# # =========================================================================
# # For a specific date folder after downloads complete:
# # current_day_dir = "./satellite_data/2024-07-15"
# # netcdf_file = os.path.join(current_day_dir, "era5_daily_2024-07-15.nc")
# # sentinel_b4 = os.path.join(current_day_dir, "sentinel2_red.tif")
# #
# # process_era5_to_match_sentinel(netcdf_file, sentinel_b4, current_day_dir)
