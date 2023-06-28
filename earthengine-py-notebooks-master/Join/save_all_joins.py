# %%
"""
<table class="ee-notebook-buttons" align="left">
    <td><a target="_blank"  href="https://github.com/giswqs/earthengine-py-notebooks/tree/master/Join/save_all_joins.ipynb"><img width=32px src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source on GitHub</a></td>
    <td><a target="_blank"  href="https://nbviewer.jupyter.org/github/giswqs/earthengine-py-notebooks/blob/master/Join/save_all_joins.ipynb"><img width=26px src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/38/Jupyter_logo.svg/883px-Jupyter_logo.svg.png" />Notebook Viewer</a></td>
    <td><a target="_blank"  href="https://colab.research.google.com/github/giswqs/earthengine-py-notebooks/blob/master/Join/save_all_joins.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" /> Run in Google Colab</a></td>
</table>
"""

# %%
"""
## Install Earth Engine API and geemap
Install the [Earth Engine Python API](https://developers.google.com/earth-engine/python_install) and [geemap](https://geemap.org). The **geemap** Python package is built upon the [ipyleaflet](https://github.com/jupyter-widgets/ipyleaflet) and [folium](https://github.com/python-visualization/folium) packages and implements several methods for interacting with Earth Engine data layers, such as `Map.addLayer()`, `Map.setCenter()`, and `Map.centerObject()`.
The following script checks if the geemap package has been installed. If not, it will install geemap, which automatically installs its [dependencies](https://github.com/giswqs/geemap#dependencies), including earthengine-api, folium, and ipyleaflet.
"""

# %%
# Installs geemap package
import subprocess

try:
    import geemap
except ImportError:
    print('Installing geemap ...')
    subprocess.check_call(["python", '-m', 'pip', 'install', 'geemap'])

# %%
import ee
import geemap

# %%
"""
## Create an interactive map 
The default basemap is `Google Maps`. [Additional basemaps](https://github.com/giswqs/geemap/blob/master/geemap/basemaps.py) can be added using the `Map.add_basemap()` function. 
"""

# %%
Map = geemap.Map(center=[40,-100], zoom=4)
Map

# %%
"""
## Add Earth Engine Python script 
"""

# %%
# Add Earth Engine dataset
# Load a primary 'collection': Landsat imagery.
primary = ee.ImageCollection('LANDSAT/LC08/C01/T1_TOA') \
    .filterDate('2014-04-01', '2014-06-01') \
    .filterBounds(ee.Geometry.Point(-122.092, 37.42))

# Load a secondary 'collection': MODIS imagery.
modSecondary = ee.ImageCollection('MODIS/006/MOD09GA') \
    .filterDate('2014-03-01', '2014-07-01')

# Define an allowable time difference: two days in milliseconds.
twoDaysMillis = 2 * 24 * 60 * 60 * 1000

# Create a time filter to define a match as overlapping timestamps.
timeFilter = ee.Filter.Or(
  ee.Filter.maxDifference(**{
    'difference': twoDaysMillis,
    'leftField': 'system:time_start',
    'rightField': 'system:time_end'
  }),
  ee.Filter.maxDifference(**{
    'difference': twoDaysMillis,
    'leftField': 'system:time_end',
    'rightField': 'system:time_start'
  })
)

# Define the join.
saveAllJoin = ee.Join.saveAll(**{
  'matchesKey': 'terra',
  'ordering': 'system:time_start',
  'ascending': True
})

# Apply the join.
landsatModis = saveAllJoin.apply(primary, modSecondary, timeFilter)

# Display the result.
print('Join.saveAll:', landsatModis.getInfo())



# %%
"""
## Display Earth Engine data layers 
"""

# %%
Map.addLayerControl() # This line is not needed for ipyleaflet-based Map.
Map