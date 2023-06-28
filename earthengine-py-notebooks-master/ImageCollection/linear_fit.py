# %%
"""
<table class="ee-notebook-buttons" align="left">
    <td><a target="_blank"  href="https://github.com/giswqs/earthengine-py-notebooks/tree/master/ImageCollection/linear_fit.ipynb"><img width=32px src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source on GitHub</a></td>
    <td><a target="_blank"  href="https://nbviewer.jupyter.org/github/giswqs/earthengine-py-notebooks/blob/master/ImageCollection/linear_fit.ipynb"><img width=26px src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/38/Jupyter_logo.svg/883px-Jupyter_logo.svg.png" />Notebook Viewer</a></td>
    <td><a target="_blank"  href="https://colab.research.google.com/github/giswqs/earthengine-py-notebooks/blob/master/ImageCollection/linear_fit.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" /> Run in Google Colab</a></td>
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
# Compute the trend of nighttime lights from DMSP.

# Add a band containing image date as years since 1990.
def createTimeBand(img):
  year = img.date().difference(ee.Date('1990-01-01'), 'year')
  return ee.Image(year).float().addBands(img)

# function createTimeBand(img) {
#   year = img.date().difference(ee.Date('1990-01-01'), 'year')
#   return ee.Image(year).float().addBands(img)
# }

# Fit a linear trend to the nighttime lights collection.
collection = ee.ImageCollection('NOAA/DMSP-OLS/CALIBRATED_LIGHTS_V4') \
    .select('avg_vis') \
    .map(createTimeBand)
fit = collection.reduce(ee.Reducer.linearFit())

# Display a single image
Map.addLayer(ee.Image(collection.select('avg_vis').first()),
         {'min': 0, 'max': 63},
         'stable lights first asset')

# Display trend in red/blue, brightness in green.
Map.setCenter(30, 45, 4)
Map.addLayer(fit,
         {'min': 0, 'max': [0.18, 20, -0.18], 'bands': ['scale', 'offset', 'scale']},
         'stable lights trend')


# %%
"""
## Display Earth Engine data layers 
"""

# %%
Map.addLayerControl() # This line is not needed for ipyleaflet-based Map.
Map