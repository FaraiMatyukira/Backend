import ee
ee.Authenticate()
ee.Initialize()
import pandas as pd
import numpy as np


def create_reduce_region_function(geometry,
                                  reducer=ee.Reducer.mean(),
                                  scale=1000,
                                  crs='EPSG:4326',
                                  bestEffort=True,
                                  maxPixels=1e13,
                                  tileScale=4):
  """Creates a region reduction function.

  Creates a region reduction function intended to be used as the input function
  to ee.ImageCollection.map() for reducing pixels intersecting a provided region
  to a statistic for each image in a collection. See ee.Image.reduceRegion()
  documentation for more details.

  Args:
    geometry:
      An ee.Geometry that defines the region over which to reduce data.
    reducer:
      Optional; An ee.Reducer that defines the reduction method.
    scale:
      Optional; A number that defines the nominal scale in meters of the
      projection to work in.
    crs:
      Optional; An ee.Projection or EPSG string ('EPSG:5070') that defines
      the projection to work in.
    bestEffort:
      Optional; A Boolean indicator for whether to use a larger scale if the
      geometry contains too many pixels at the given scale for the operation
      to succeed.
    maxPixels:
      Optional; A number specifying the maximum number of pixels to reduce.
    tileScale:
      Optional; A number representing the scaling factor used to reduce
      aggregation tile size; using a larger tileScale (e.g. 2 or 4) may enable
      computations that run out of memory with the default.

  Returns:
    A function that accepts an ee.Image and reduces it by region, according to
    the provided arguments.
  """

  def reduce_region_function(img):
    """Applies the ee.Image.reduceRegion() method.

    Args:
      img:
        An ee.Image to reduce to a statistic by region.

    Returns:
      An ee.Feature that contains properties representing the image region
      reduction results per band and the image timestamp formatted as
      milliseconds from Unix epoch (included to enable time series plotting).
    """

    stat = img.reduceRegion(
        reducer=reducer,
        geometry=geometry,
        scale=scale,
        crs=crs,
        bestEffort=bestEffort,
        maxPixels=maxPixels,
        tileScale=tileScale)

    return ee.Feature(geometry, stat).set({'millis': img.date().millis()})
  return reduce_region_function

def fc_to_dict(fc):
  prop_names = fc.first().propertyNames()
  prop_lists = fc.reduceColumns(
      reducer=ee.Reducer.toList().repeat(prop_names.size()),
      selectors=prop_names).get('list')

  return ee.Dictionary.fromLists(prop_names, prop_lists)

def add_date_info(df):
  df['Timestamp'] = pd.to_datetime(df['millis'], unit='ms')
  df['Year'] = pd.DatetimeIndex(df['Timestamp']).year
  df['Month'] = pd.DatetimeIndex(df['Timestamp']).month
  df['Day'] = pd.DatetimeIndex(df['Timestamp']).day
  df['DOY'] = pd.DatetimeIndex(df['Timestamp']).dayofyear
  return df

class Gather:
    def gatherdataprofile(self,polygon):
        today = ee.Date(pd.to_datetime('today'))
        date_range = ee.DateRange(today.advance(-100, 'years'), today)
        pdsi = ee.ImageCollection('GRIDMET/DROUGHT').filterDate(date_range).select('pdsi')
        aoi = ee.FeatureCollection('EPA/Ecoregions/2013/L3').filter(
            ee.Filter.eq('na_l3name', 'Sierra Nevada')).geometry()

        ndvi = ee.ImageCollection('MODIS/061/MOD13A2').filterDate(date_range).select('NDVI', 'EVI','sur_refl_b01', 'sur_refl_b02', 'sur_refl_b03', 'sur_refl_b07')
        aoi  = ee.Geometry.Polygon([   [20.391593724627356,-26.966691382307616]
        ,[20.402365476031164,-26.966691382307616], [20.402365476031164,-26.961068613709564]
        ,[20.391593724627356,-26.961068613709564]
        , [20.391593724627356,-26.966691382307616]])

        reduce_ndvi = create_reduce_region_function(
            geometry=aoi, reducer=ee.Reducer.mean(), scale=1000, crs='EPSG:3310')

        ndvi_stat_fc = ee.FeatureCollection(ndvi.map(reduce_ndvi)).filter(
            ee.Filter.notNull(ndvi.first().bandNames()))

        ndvi_dict = fc_to_dict(ndvi_stat_fc).getInfo()
        ndvi_df = pd.DataFrame(ndvi_dict)

        ndvi_df['NDVI'] = ndvi_df['NDVI'] / 10000
        ndvi_df  = add_date_info(ndvi_df)

        ndvi_df.to_csv("Potch farm data.csv",mode='a',sep =";")