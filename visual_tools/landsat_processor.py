import ee
import geemap
import geopandas as gpd
import ipywidgets as widgets
from IPython.display import display
from ipyleaflet import WidgetControl
import os


def authenticate(ee_project_name="driven-atrium-401315"):
    try:
        ee.Initialize(project=ee_project_name)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=ee_project_name)


authenticate()


def roi_to_geometry(roi_def):
    '''
    Convert `roi` to an ee.Geometry.

    Accepts:
      - ee.Geometry / ee.Feature / ee.FeatureCollection  -> returned as geometry
      - str / os.PathLike: a file path to a vector file
          * .shp  (also .zip shapefile)
          * .geojson / .json
          * .kml
          * .gpkg
          * .gdb  (file geodatabase)
      - anything else that exposes a .geometry() method

    The result is always a single ee.Geometry. Multi-feature files are
    dissolved into one geometry (union)
    '''
    if isinstance(roi_def, (ee.Geometry, ee.Feature, ee.FeatureCollection)):
        if isinstance(roi_def, ee.Geometry):
            return roi_def
        if isinstance(roi_def, ee.Feature):
            return roi_def.geometry()
        return roi_def.geometry()  # FeatureCollection -> .geometry() unions all

    if not isinstance(roi_def, (str, os.PathLike)):
        raise TypeError(
            f"`roi` must be an ee.Geometry or a path string. Got: {type(roi_def)}"
        )

    path = os.fspath(roi_def)
    if not os.path.exists(path):
        raise FileNotFoundError(f"ROI file not found: {path}")

    ext = os.path.splitext(path)[1].lower()

    if ext in ('.geojson', '.json'):
        fc = geemap.geojson_to_ee(path)
        return fc.geometry()

    if ext == '.kml':
        fc = geemap.kml_to_ee(path)
        return fc.geometry()

    if ext in ('.shp', '.zip'):
        fc = geemap.shp_to_ee(path)
        return fc.geometry()

    try:
        gdf = gpd.read_file(path)

        if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)


        if len(gdf) > 1:
            gdf = gdf.dissolve()

        geojson_str = gdf.geometry.iloc[0].__geo_interface__
        return ee.Geometry(geojson_str, geodesic=False)

    except ImportError as e:
        raise ImportError(
            f"Format '{ext}' requires geopandas. Install with `pip install geopandas`."
        ) from e
    except Exception as e:
        raise RuntimeError(f"Could not read ROI file '{path}': {e}") from e



def process_landsat(image):
    qa = image.select('QA_PIXEL')
    cloud_shadow_mask = qa.bitwiseAnd(1 << 4).eq(0)
    cloud_mask = qa.bitwiseAnd(1 << 3).eq(0)
    dilated_mask = qa.bitwiseAnd(1 << 1).eq(0)
    valid_mask = cloud_shadow_mask.And(cloud_mask).And(dilated_mask)

    st_kelvin = image.select('ST_B10').multiply(0.00341802).add(149.0)
    st_celsius = st_kelvin.subtract(273.15)

    return (
        st_celsius
        .updateMask(valid_mask)
        .rename('LST_C')
        .copyProperties(image, ['system:time_start'])
    )


def custom_reducer(collection, start, end, agg_stat):
    if agg_stat == "mean":
        img = collection.filterDate(start, end).mean()
    elif agg_stat == "median":
        img = collection.filterDate(start, end).median()
    elif agg_stat == "max":
        img = collection.filterDate(start, end).max()
    elif agg_stat == "min":
        img = collection.filterDate(start, end).min()
    elif agg_stat == "stdv":
        img = collection.filterDate(start, end).reduce(ee.Reducer.stdDev())
    else:
        raise ValueError(f"Provided agg_stat: {agg_stat} is not defined!")
    return img.set('system:time_start', start.millis())


def collapse_to_daily_mosaic(collection):
    def add_day_tag(img):
        date_str = ee.Date(img.get('system:time_start')).format('YYYY-MM-dd')
        return img.set('day_millis', ee.Date(date_str).millis())

    tagged_col = ee.ImageCollection(collection).map(add_day_tag)
    distinct_days = tagged_col.aggregate_array('day_millis').distinct()

    def mosaic_per_day(day_millis):
        d = ee.Date(day_millis)
        return (
            tagged_col.filterDate(d, d.advance(1, 'day'))
            .mosaic()
            .set('system:time_start', d.millis())
        )

    return ee.ImageCollection(distinct_days.map(mosaic_per_day))


def aggregate_time_series(collection, start, end, freq, agg_stat):
    startDate = ee.Date(start)
    endDate = ee.Date(end)

    if freq == 'monthly':
        n = endDate.difference(startDate, 'month').round()
        def f(index):
            s = startDate.advance(ee.Number(index), 'month')
            return custom_reducer(collection, s, s.advance(1, 'month'), agg_stat)
    elif freq == 'weekly':
        n = endDate.difference(startDate, 'week').round()
        def f(index):
            s = startDate.advance(ee.Number(index).multiply(7), 'day')
            return custom_reducer(collection, s, s.advance(7, 'day'), agg_stat)
    elif freq == '8-day':
        n = endDate.difference(startDate, 'day').divide(8).round()
        def f(index):
            s = startDate.advance(ee.Number(index).multiply(8), 'day')
            return custom_reducer(collection, s, s.advance(8, 'day'), agg_stat)
    else:
        n = endDate.difference(startDate, 'day').divide(freq).round()
        def f(index):
            s = startDate.advance(ee.Number(index).multiply(freq), 'day')
            return custom_reducer(collection, s, s.advance(freq, 'day'), agg_stat)

    return ee.ImageCollection(ee.List.sequence(0, n.subtract(1)).map(f))


def exceed_collection_func(collection, threshold):
    exceed = collection.map(
        lambda img: img.gt(threshold).copyProperties(img, ['system:time_start'])
    )
    return exceed.sum().rename('days_above_threshold').toFloat()


# ---------------------------------------------------------------- #
# Dynamic colorbar helpers
# ---------------------------------------------------------------- #

def _remove_colorbar(Map):
    '''
    Remove any existing colorbar from a geemap.Map across geemap versions.

    geemap stores the colorbar in one (or more) of these places depending
    on version:
      - Map.colorbar                         (attribute, geemap >= 0.30)
      - Map._colorbar                        (attribute, some versions)
      - A WidgetControl with name 'colorbar' (older versions)
      - Entries in Map.controls whose widget has an 'markdown' / 'colorbar' child
    '''
    # 1) Attribute-based storage (most common in modern geemap)
    for attr in ('colorbar', '_colorbar'):
        ctrl = getattr(Map, attr, None)
        if ctrl is not None:
            try:
                Map.remove_control(ctrl)
            except Exception:
                pass
            try:
                setattr(Map, attr, None)
            except Exception:
                pass

    # 2) Control-based storage — remove anything that looks like a colorbar
    to_remove = []
    for ctrl in list(Map.controls):
        name = getattr(ctrl, 'name', '') or ''
        if name.lower() == 'colorbar':
            to_remove.append(ctrl)
            continue
        # Some geemap versions wrap the colorbar inside a WidgetControl
        # whose .widget holds the actual ipywidgets.HTML / VBox
        widget = getattr(ctrl, 'widget', None)
        if widget is not None:
            widget_type = type(widget).__name__.lower()
            if 'colorbar' in widget_type:
                to_remove.append(ctrl)
                continue
            # Match by children or by known colorbar marker strings
            try:
                children = getattr(widget, 'children', ())
            except Exception:
                children = ()
            for child in children:
                child_cls = type(child).__name__.lower()
                if 'colorbar' in child_cls:
                    to_remove.append(ctrl)
                    break

    for ctrl in to_remove:
        try:
            Map.remove_control(ctrl)
        except Exception:
            pass


def _remove_legend(Map):
    '''Remove any existing geemap legend, across storage locations.'''
    for attr in ('legend', '_legend'):
        ctrl = getattr(Map, attr, None)
        if ctrl is not None:
            try:
                Map.remove_control(ctrl)
            except Exception:
                pass
            try:
                setattr(Map, attr, None)
            except Exception:
                pass

    to_remove = []
    for ctrl in list(Map.controls):
        name = (getattr(ctrl, 'name', '') or '').lower()
        if name in ('legend', 'colorbar'):
            to_remove.append(ctrl)
            continue
        widget = getattr(ctrl, 'widget', None)
        if widget is not None and 'legend' in type(widget).__name__.lower():
            to_remove.append(ctrl)

    for ctrl in to_remove:
        try:
            Map.remove_control(ctrl)
        except Exception:
            pass

def set_colorbar_mode(
    Map,
    mode,
    lst_viz,
    exceed_viz,
    temp_threshold
):
    '''
    Swap the colorbar (and legend) on the map to match the active layer mode.

    Parameters
    ----------
    mode : {'lst', 'exceedance'}
    '''
    _remove_colorbar(Map)
    _remove_legend(Map)

    if mode == 'lst':
        Map.add_colorbar(
            colors=lst_viz['palette'],
            vmin=lst_viz['min'],
            vmax=lst_viz['max'],
            label="Land Surface Temperature (°C)",
        )
    elif mode == 'exceedance':
        Map.add_colorbar(
            colors=exceed_viz['palette'],
            vmin=exceed_viz['min'],
            vmax=exceed_viz['max'],
            label=f" Days LST > {int(temp_threshold)} °C",
        )
    else:
        raise ValueError(f"Unknown colorbar mode: {mode}")


# ---------------------------------------------------------------- #
# Main
# ---------------------------------------------------------------- #

def automate_landsat(
    roi,
    start_date='2021-01-01',
    end_date='2021-12-30',
    aggregation_time='monthly',
    aggregation_stat="max",
    temp_threshold=30.0):
    
    #------ Read AOI file -------
    roi_geom = roi_to_geometry(roi_def=roi)
    # ---- Data prep -------------------------------------------------
    l8 = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
          .filterDate(start_date, end_date).filterBounds(roi_geom))
    l9 = (ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
          .filterDate(start_date, end_date).filterBounds(roi_geom))
    landsat_collection = l8.merge(l9)

    processed = collapse_to_daily_mosaic(landsat_collection.map(process_landsat))
    aggregated_stack = aggregate_time_series(
        processed, start_date, end_date, aggregation_time, aggregation_stat
    )

    stack_size = aggregated_stack.size().getInfo()
    print(f"Stack created ({aggregation_time}, {aggregation_stat}). "
          f"Frames: {stack_size}")

    days_exceeding_image = exceed_collection_func(aggregated_stack, temp_threshold)

    # ---- Viz params ------------------------------------------------
    lst_viz = {
        'min': 0.0, 'max': 50.0,
        'palette': ['blue', 'cyan', 'green', 'yellow', 'orange', 'red'],
    }
    exceed_viz = {
        'min': 0, 'max': 120,
        'palette': ['lightblue', 'yellow', 'orange', 'red', 'darkred'],
    }
    

    # ---- Map setup -------------------------------------------------
    roi_centroid = roi_geom.centroid().coordinates().getInfo()
    Map = geemap.Map(center=[roi_centroid[1], roi_centroid[0]], zoom=10)

    Map.addLayer(roi_geom, {'color': 'black', 'fillColor': '00000000'},
                 'Region of Interest')
    Map.addLayer(aggregated_stack.mean().clip(roi_geom), lst_viz,
                 f'Mean LST ({aggregation_time})')
    Map.addLayer(days_exceeding_image.clip(roi_geom), exceed_viz,
                 f'Total Days > {temp_threshold}°C', shown=False)

    # ---- Time-series frames ---------------------------------------
    image_list = aggregated_stack.toList(stack_size)
    date_options = []
    for i in range(stack_size):
        img = ee.Image(image_list.get(i))
        date_str = (ee.Date(img.get('system:time_start'))
                    .format('YYYY-MM-dd').getInfo())
        date_options.append((date_str, i))

    SELECTED_LAYER_NAME = 'Selected Frame'

    def remove_selected_layer():
        for layer in list(Map.layers):
            if layer.name == SELECTED_LAYER_NAME:
                Map.remove_layer(layer)
                break

    def show_frame(idx):
        remove_selected_layer()
        img = ee.Image(image_list.get(int(idx)))
        Map.addLayer(img, lst_viz, SELECTED_LAYER_NAME)

    def update_selected_layer(change):
        if change['type'] == 'change' and change['name'] == 'value':
            show_frame(change['new'])

    dropdown = widgets.Dropdown(
        options=date_options,
        description='Time Frame:',
        style={'description_width': 'initial'},
    )
    dropdown.observe(update_selected_layer, names='value')

    if date_options:
        show_frame(dropdown.value)

    # ---- Colorbar mode switcher -----------------------------------
    colorbar_mode = widgets.ToggleButtons(
        options=[('LST (°C)', 'lst'),
                 (f'Days > {int(temp_threshold)}°C', 'exceedance')],
        value='lst',
        description='Colorbar:',
        style={'description_width': 'initial'},
    )

    def on_mode_change(change):
        if change['type'] == 'change' and change['name'] == 'value':
            set_colorbar_mode(
                Map,
                change['new'],
                lst_viz=lst_viz,
                exceed_viz=exceed_viz,
                temp_threshold=temp_threshold
            )

    colorbar_mode.observe(on_mode_change, names='value')

    # Initial colorbar
    set_colorbar_mode(
        Map, colorbar_mode.value,
        lst_viz=lst_viz, exceed_viz=exceed_viz,
        temp_threshold=temp_threshold
    )

    # ---- Export UI -------------------------------------------------
    export_target_dropdown = widgets.Dropdown(
        options=[
            ('Selected Time-Series Frame', 'frame'),
            ('Mean LST Composite', 'mean'),
            ('Threshold Exceedance Count', 'exceedance'),
            ('All Stack Frames (Batch Export)', 'all_stack'),
        ],
        value='frame',
        description='Export Item:',
        style={'description_width': 'initial'},
    )

    def on_export_clicked(b):
        target = export_target_dropdown.value

        def _export(image, filename):
            task = ee.batch.Export.image.toDrive(
                image=image.clip(roi_geom),
                description=filename,
                folder='EarthEngine_Exports',
                fileNamePrefix=filename,
                region=roi_geom,
                scale=30,
                maxPixels=1e9,
            )
            task.start()
            print(f"Export task started for [{filename}] — check GEE Tasks.")

        if target == 'frame':
            idx = int(dropdown.value)
            date_str = date_options[idx][0]
            filename = f"Landsat_LST_{aggregation_time}_{aggregation_stat}_{date_str}"
            _export(ee.Image(image_list.get(idx)), filename)

        elif target == 'mean':
            filename = f"Landsat_LST_Mean_{aggregation_time}_{start_date}_to_{end_date}"
            _export(aggregated_stack.mean(), filename)

        elif target == 'exceedance':
            filename = f"Landsat_Days_Above_{temp_threshold}C_{start_date}_to_{end_date}"
            _export(days_exceeding_image, filename)

        elif target == 'all_stack':
            print(f"Initiating batch export for all {stack_size} frames...")
            for i, (date_str, _) in enumerate(date_options):
                filename = f"Landsat_LST_{aggregation_time}_{aggregation_stat}_{date_str}"
                _export(ee.Image(image_list.get(i)), filename)
            print(f"Submitted all {stack_size} tasks — check GEE Tasks.")

    export_button = widgets.Button(
        description='Export to Google Drive',
        button_style='success',
        icon='download',
    )
    export_button.on_click(on_export_clicked)

    # ---- Control panel --------------------------------------------
    control_panel = widgets.VBox([
        widgets.HTML("<b>Landsat LST Stack Explorer</b>"),
        widgets.HTML("<hr style='margin: 4px 0;'>"),
        widgets.HTML("<b>1. Preview Time Frame:</b>"),
        dropdown,
        widgets.HTML("<b>Colorbar Mode:</b>"),
        colorbar_mode,
        widgets.HTML("<hr style='margin: 4px 0;'>"),
        widgets.HTML("<b>2. Export Settings:</b>"),
        export_target_dropdown,
        export_button,
    ])
    Map.add_control(WidgetControl(widget=control_panel, position='bottomleft'))

    display(Map)
    return Map, dropdown, colorbar_mode, aggregated_stack
