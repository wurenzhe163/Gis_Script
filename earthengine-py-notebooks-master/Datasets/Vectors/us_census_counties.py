# %%
"""
<table class="ee-notebook-buttons" align="left">
    <td><a target="_blank"  href="https://github.com/giswqs/earthengine-py-notebooks/tree/master/Datasets/Vectors/us_census_counties.ipynb"><img width=32px src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source on GitHub</a></td>
    <td><a target="_blank"  href="https://nbviewer.jupyter.org/github/giswqs/earthengine-py-notebooks/blob/master/Datasets/Vectors/us_census_counties.ipynb"><img width=26px src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/38/Jupyter_logo.svg/883px-Jupyter_logo.svg.png" />Notebook Viewer</a></td>
    <td><a target="_blank"  href="https://colab.research.google.com/github/giswqs/earthengine-py-notebooks/blob/master/Datasets/Vectors/us_census_counties.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" /> Run in Google Colab</a></td>
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
Map.setCenter(-110, 40, 5)
states = ee.FeatureCollection('TIGER/2018/States')
    # .filter(ee.Filter.eq('STUSPS', 'MN'))
# // Turn the strings into numbers
states = states.map(lambda f: f.set('STATEFP', ee.Number.parse(f.get('STATEFP'))))

state_image = ee.Image().float().paint(states, 'STATEFP')

visParams = {
  'palette': ['purple', 'blue', 'green', 'yellow', 'orange', 'red'],
  'min': 0,
  'max': 50,
  'opacity': 0.8,
};

counties = ee.FeatureCollection('TIGER/2016/Counties')  

image = ee.Image().paint(states, 0, 2)
Map.setCenter(-99.844, 37.649, 5)
# Map.addLayer(image, {'palette': 'FF0000'}, 'TIGER/2018/States')
Map.addLayer(image, visParams, 'TIGER/2016/States');
Map.addLayer(ee.Image().paint(counties, 0, 1), {}, 'TIGER/2016/Counties')


# %%
"""
## Display Earth Engine data layers 
"""

# %%
Map.addLayerControl() # This line is not needed for ipyleaflet-based Map.
Map