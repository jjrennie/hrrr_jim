########################################
# Written by Jared Rennie
# Derived from Kevin Tyle
# https://github.com/ktyle/python_pangeo_ams2021/blob/main/notebooks/03_Pangeo_HRRR.ipynb
########################################

# Import Packages
import sys, time, datetime, os
import numpy as np
import geopandas as gpd
import pandas as pd
import xarray as xr
import s3fs
import metpy

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader

# Declare directories
main_directory='.'
plot_directory=main_directory+'/results_JIM'

# Define Plotting Info/Bounds
dpi=300
plt.style.use('dark_background')
minLat = 22    
maxLat = 50   
minLon = -120 
maxLon = -73 

####################
# BEGIN PROGRAM
####################
start=time.time()

# Read in Arguments 
if len(sys.argv) < 5:
    sys.exit("USAGE: <YYYY> <MM> <DD> <HH> <ELEMENT>\nExample: ipython plot_hrrr_JIM.py 2021 08 11 12")

# Get Date Info
year= "%04i" % int(sys.argv[1])
month= "%02i" % int(sys.argv[2])
day= "%02i" % int(sys.argv[3])
hour = "%02i" % int(sys.argv[4])
element= 'Td'
date = year+month+day
init_name=year+month+day+': '+hour+'UTC'
init_time=year+month+day+hour
print("INIT: ",init_name)

# Get Element Info
var = 'DPT'
level = '2m_above_ground'
unit='$^\circ$F'
color_map='BrBG' 
outElement='2m Dew Point'
bounds=np.arange(40,100,5)

####################
# PART ONE: Analysis
#  i.e. T=0
####################
# Access HRRR from AWS ... projection dimensions are in url2
# Note That 'anl' stands for Analysis (Initialization)
fs = s3fs.S3FileSystem(anon=True)
url1 = 's3://hrrrzarr/sfc/' + date + '/' + date + '_' + hour + 'z_anl.zarr/' + level + '/' + var + '/' + level
url2 = 's3://hrrrzarr/sfc/' + date + '/' + date + '_' + hour + 'z_anl.zarr/' + level + '/' + var

file1 = s3fs.S3Map(url1, s3=fs)
file2 = s3fs.S3Map(url2, s3=fs)

# Open Datasets
print("\nOPEN ANALYSIS DATASETS")
print ("\t",url1)
print ("\t",url2)
ds = xr.open_mfdataset([file1,file2], engine='zarr')

# Define HRRR projection
lon1 = -97.5
lat1 = 38.5
slat = 38.5
projData= ccrs.LambertConformal(central_longitude=lon1,central_latitude=lat1,standard_parallels=[slat])

# Get Temperature Data, and create the JIM
airTemp = ds[var].metpy.convert_units('degF')
JIM=np.where(airTemp.values >= 55, 1, 0)

# PLOT
print("\nPLOT ANALYSIS")
dpi=300
plt.style.use('dark_background')

# Get Plotting Coordinates
x = airTemp.projection_x_coordinate
y = airTemp.projection_y_coordinate

# Set Up Figure
fig= plt.figure(num=1, figsize=(8,5), dpi=dpi, facecolor='w', edgecolor='k')
ax = fig.add_axes([0, 0, 1, 1], projection=projData)
ax.set_extent([minLon, maxLon, minLat, maxLat], crs=ccrs.PlateCarree())
ax.set_facecolor('black')
ax.add_feature(cfeature.COASTLINE.with_scale('50m'),linewidth=0.5)
ax.add_feature(cfeature.BORDERS.with_scale('50m'),linewidth=0.5)
ax.add_feature(cfeature.STATES.with_scale('50m'),linewidth=0.5)

# Plot Data
cmap = mcolors.ListedColormap(['#66c2a5','#fc8d62'])
ax.pcolormesh(x, y, JIM,cmap=cmap)

# Add Colormap
bounds=np.array([0,1,2],dtype='i')
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
cax = fig.add_axes([0.1, -0.035, 0.8, 0.03])
cbar=plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm),cax=cax,boundaries=bounds,extend='neither',extendfrac='auto',ticks=bounds,spacing='uniform',orientation='horizontal')
cbar.ax.tick_params(labelsize=10)

# Fix Tick Marks
loc = bounds + .5
cbar.set_ticks(loc)
labels=np.array(['Not Miserable (Td < 55$^\circ$F)','Miserable (Td >= 55$^\circ$F)'],dtype='str')
cbar.set_ticklabels(labels)

# Add Titles
plt.suptitle('Where Jared is Miserable on '+init_name,fontsize=17,y=1.05)
plt.annotate('Source: HRRR, init= '+init_name+'\nMade By Jared Rennie @jjrennie',xy=(1.045, -3.51), xycoords='axes fraction', fontsize=5,backgroundcolor='black',color='white',horizontalalignment='right', verticalalignment='bottom')

# Save Figure
plt.savefig(plot_directory+"/hrrr_jim_init-"+init_time+"-anls-00.png",bbox_inches='tight') 
plt.clf()
plt.close()

####################
# PART TWO: Forecast
####################
# Access HRRR from AWS ... projection dimensions are in url2
# Note That 'fcst' stands for Forecast 
fs = s3fs.S3FileSystem(anon=True)
url1 = 's3://hrrrzarr/sfc/' + date + '/' + date + '_' + hour + 'z_fcst.zarr/' + level + '/' + var + '/' + level
url2 = 's3://hrrrzarr/sfc/' + date + '/' + date + '_' + hour + 'z_fcst.zarr/' + level + '/' + var

file1 = s3fs.S3Map(url1, s3=fs)
file2 = s3fs.S3Map(url2, s3=fs)

# Open Datasets
print("\nOPEN FORECAST DATASETS")
print ("\t",url1)
print ("\t",url2)
ds = xr.open_mfdataset([file1,file2], engine='zarr')

# Define HRRR projection
lon1 = -97.5
lat1 = 38.5
slat = 38.5
projData= ccrs.LambertConformal(central_longitude=lon1,central_latitude=lat1,standard_parallels=[slat])

# Plot Each Forecast Hour
print("\nPLOT FORECASTS")
dpi=300
plt.style.use('dark_background')

# Set Bounds
minLat = 22    
maxLat = 50   
minLon = -120 
maxLon = -73 

init_name=year+month+day+': '+hour+'UTC'
init_time=year+month+day+hour

for time_counter in range(0,len(ds.time.values)):
    # Organize Forecast Time
    fcstTime=ds.time.values[time_counter]
    fcstYear= "%04i" % int(fcstTime.astype('datetime64[h]').astype(str)[0:4])
    fcstMonth="%02i" % int(fcstTime.astype('datetime64[h]').astype(str)[5:7])
    fcstDay="%02i" % int(fcstTime.astype('datetime64[h]').astype(str)[8:10])
    fcstHour="%02i" % int(fcstTime.astype('datetime64[h]').astype(str)[11:13])
    
    forecast_name=fcstYear+fcstMonth+fcstDay+': '+fcstHour+'UTC'
    forecast_time=fcstYear+fcstMonth+fcstDay+fcstHour
    forecast_hour= "%02i" % (time_counter+1)
    print("\tT= ",forecast_hour,": ",forecast_name)

    # Get Temperature Data
    airTemp = ds[var][time_counter].metpy.convert_units('degF')
    JIM=np.where(airTemp.values >= 55, 1, 0)

    # Get Plotting Coordinates
    x = airTemp.projection_x_coordinate
    y = airTemp.projection_y_coordinate

    # Set Up Figure
    fig= plt.figure(num=1, figsize=(8,5), dpi=dpi, facecolor='w', edgecolor='k')
    ax = fig.add_axes([0, 0, 1, 1], projection=projData)
    ax.set_extent([minLon, maxLon, minLat, maxLat], crs=ccrs.PlateCarree())
    ax.set_facecolor('black')
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'),linewidth=0.5)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'),linewidth=0.5)
    ax.add_feature(cfeature.STATES.with_scale('50m'),linewidth=0.5)

    # Plot Data
    cmap = mcolors.ListedColormap(['#66c2a5','#fc8d62'])
    ax.pcolormesh(x, y, JIM,cmap=cmap)

    # Add Colormap
    bounds=np.array([0,1,2],dtype='i')
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    cax = fig.add_axes([0.1, -0.035, 0.8, 0.03])
    cbar=plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm),cax=cax,boundaries=bounds,extend='neither',extendfrac='auto',ticks=bounds,spacing='uniform',orientation='horizontal')
    cbar.ax.tick_params(labelsize=10)

    # Fix Tick Marks
    loc = bounds + .5
    cbar.set_ticks(loc)
    labels=np.array(['Not Miserable (Td < 55$^\circ$F)','Miserable (Td >= 55$^\circ$F)'],dtype='str')
    cbar.set_ticklabels(labels)

    # Add Titles
    plt.suptitle('Where Jared is Miserable on '+forecast_name,fontsize=17,y=1.05)
    plt.annotate('Source: HRRR, init= '+init_name+' | fcst= '+forecast_name+'\nMade By Jared Rennie @jjrennie',xy=(1.045, -3.51), xycoords='axes fraction', fontsize=5,backgroundcolor='black',color='white',horizontalalignment='right', verticalalignment='bottom')

    # Save Figure
    plt.savefig(plot_directory+"/hrrr_jim_init-"+init_time+"-fcst-"+forecast_hour+".png",bbox_inches='tight') 
    plt.clf()
    plt.close()

####################
# DONE
####################
print("DONE!")
end=time.time()
print ("Runtime: %8.1f seconds." % (end-start))
sys.exit()
