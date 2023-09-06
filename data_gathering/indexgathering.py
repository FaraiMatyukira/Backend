#!/usr/bin/env python
# coding: utf-8

# In[1]:


import ee
ee.Authenticate()
ee.Initialize()


# In[2]:


import pandas as pd
import altair as alt
import numpy as np
import folium


# In[3]:


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


# In[4]:


def fc_to_dict(fc):
  prop_names = fc.first().propertyNames()
  prop_lists = fc.reduceColumns(
      reducer=ee.Reducer.toList().repeat(prop_names.size()),
      selectors=prop_names).get('list')

  return ee.Dictionary.fromLists(prop_names, prop_lists)


# In[5]:


def add_date_info(df):
  df['Timestamp'] = pd.to_datetime(df['millis'], unit='ms')
  df['Year'] = pd.DatetimeIndex(df['Timestamp']).year
  df['Month'] = pd.DatetimeIndex(df['Timestamp']).month
  df['Day'] = pd.DatetimeIndex(df['Timestamp']).day
  df['DOY'] = pd.DatetimeIndex(df['Timestamp']).dayofyear
  return df


# In[6]:


today = ee.Date(pd.to_datetime('today'))
date_range = ee.DateRange(today.advance(-100, 'years'), today)
pdsi = ee.ImageCollection('GRIDMET/DROUGHT').filterDate(date_range).select('pdsi')
aoi = ee.FeatureCollection('EPA/Ecoregions/2013/L3').filter(
    ee.Filter.eq('na_l3name', 'Sierra Nevada')).geometry()


# In[ ]:


class Home :

    def say (self):
        print("hello")


# In[31]:


ndvi = ee.ImageCollection('MODIS/061/MOD13A2').filterDate(date_range).select('NDVI', 'EVI','sur_refl_b01', 'sur_refl_b02', 'sur_refl_b03', 'sur_refl_b07')
aoi  = ee.Geometry.Polygon([   [20.391593724627356,-26.966691382307616]
,[20.402365476031164,-26.966691382307616], [20.402365476031164,-26.961068613709564]
,[20.391593724627356,-26.961068613709564]
, [20.391593724627356,-26.966691382307616]])

reduce_ndvi = create_reduce_region_function(
    geometry=aoi, reducer=ee.Reducer.mean(), scale=1000, crs='EPSG:3310')

ndvi_stat_fc = ee.FeatureCollection(ndvi.map(reduce_ndvi)).filter(
    ee.Filter.notNull(ndvi.first().bandNames()))


# In[8]:


reduce_pdsi = create_reduce_region_function(
    geometry=aoi, reducer=ee.Reducer.mean(), scale=5000, crs='EPSG:3310')

pdsi_stat_fc = ee.FeatureCollection(pdsi.map(reduce_pdsi)).filter(
    ee.Filter.notNull(pdsi.first().bandNames()))


# In[25]:


# pdsi_dict = fc_to_dict(pdsi_stat_fc).getInfo()


# In[26]:


# pdsi_df = pd.DataFrame(pdsi_dict)


# In[27]:


# display(pdsi_df)
# print(pdsi_df.dtypes)


# In[32]:


ndvi_dict = fc_to_dict(ndvi_stat_fc).getInfo()
ndvi_df = pd.DataFrame(ndvi_dict)

ndvi_df['NDVI'] = ndvi_df['NDVI'] / 10000
ndvi_df  = add_date_info(ndvi_df)


display(ndvi_df)
ndvi_df.to_csv("Potch farm data.csv",mode='a',sep =";")


# In[29]:



count=0    
for i,value in ndvi_df.iterrows():
    if value["C"] == 1 :
        count+=1
        # print(value)
print(count)


# In[ ]:


ndvi_doy_range = [224, 272]

ndvi_df_sub = ndvi_df[(ndvi_df['DOY'] >= ndvi_doy_range[0])
                      & (ndvi_df['DOY'] <= ndvi_doy_range[1])]

# ndvi_df_sub = ndvi_df_sub.groupby('Year').agg('min')


# In[ ]:


pdsi_df = add_date_info(pdsi_df)
display(pdsi_df)


# In[ ]:


pdsi_doy_range = [1, 272]

pdsi_df_sub = pdsi_df[(pdsi_df['DOY'] >= pdsi_doy_range[0])
                      & (pdsi_df['DOY'] <= pdsi_doy_range[1])]

# pdsi_df_sub = pdsi_df_sub.groupby('Year').agg('mean')


# In[ ]:


ndvi_pdsi_df = pd.merge(
    ndvi_df_sub, pdsi_df_sub, how='left', on='Month').reset_index()

ndvi_pdsi_df = ndvi_pdsi_df[['Year','NDVI', 'pdsi']]


ndvi_pdsi_df.head(5)


# In[ ]:


def ndvi_class(row):
    if row['NDVI'] > 0.2 and row['NDVI'] > 1:
        val = 1
    else:
        print(type(row['NDVI']))
        val = 0
    return val

ndvi_pdsi_df['C'] = ndvi_pdsi_df.apply(ndvi_class, axis=1)
ndvi_pdsi_df.head()

