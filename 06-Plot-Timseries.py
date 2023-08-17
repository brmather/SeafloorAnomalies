#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import gplately
import pandas as pd
from scipy import ndimage
from matplotlib.lines import Line2D
from skimage import feature

# agegrid_dir = "/Users/ben/Dropbox/USyd/GPlates/"
# agegrid_filename = agegrid_dir+"SampleData/1Ga_model/v2/AgeGrids_0.5d/masked/seafloor_age_mask_{:.1f}Ma.nc"

# reconstruction_times = np.arange(0,1001)

# Call GPlately's DataServer from the download.py module
# gdownload = gplately.download.DataServer("Merdith2021")

agegrid_dir = "/Users/ben/Dropbox/USyd/GPlates/"
agegrid_filename = agegrid_dir+"slab_dip/Clennet_AgeGrids_0.1d_masked/seafloor_age_mask_{:.1f}Ma.nc"
LIP_filename = "/Users/ben/Dropbox/USyd/GPlates/SampleData/FeatureCollections/LargeIgneousProvinces_VolcanicProvinces/Johansson_etal_2018_VolcanicProvinces/Johansson_etal_2018_VolcanicProvinces_v2.gpmlz"

reconstruction_times = np.arange(0,171)

# Call GPlately's DataServer from the download.py module
gdownload = gplately.download.DataServer("Clennett2020")



# Obtain all rotation files, topology features and static polygons from Muller et al. 2019
rotation_model, topology_features, static_polygons = gdownload.get_plate_reconstruction_files()


# In[4]:


model = gplately.PlateReconstruction(rotation_model, topology_features, static_polygons)

# Obtain geometry shapefiles with gdownload
coastlines, continents, COBs = gdownload.get_topology_geometries()

# Call the PlotTopologies object
gplot = gplately.plot.PlotTopologies(model, coastlines, continents, COBs)



# In[5]:


metal_dict = dict()

commodities = ['Cu (Mt)', 'Pb (Mt)', 'Zn (Mt)', 'Ni (Mt)']
sheets = ['PbZn-CD', 'PbZn-MVT', 'Cu-sed', 'Magmatic Ni', 'VMS', 'Cu-por', 'IOCG']

for sheet in sheets:
    df = pd.read_excel('data/base_metal_deposit_compilation.xls', sheet_name=sheet, na_values='ND')
    df = df[df['Age (Ga)'].notna()]
    df = df[df['Age (Ga)']*1000 <= reconstruction_times.max()]

    if df.shape[0] > 0:
        metal_dict[sheet] = df
    else:
        sheets.remove(sheet)
        
symbols = ['o', 'v', 's', '*', 'd', '^', 'P']*2


# In[6]:


pts_dict = dict()

for i, sheet in enumerate(sheets):
    df = metal_dict[sheet]

    pts_dict[sheet] = gplately.Points(model, df['Lon'], df['Lat'])


# In[17]:

def grad(raster, tol_grad=2, iter_dilation=0, mask=True, return_gradient=False):
    image = raster.fill_NaNs(return_array=True)
    gradX, gradY = np.gradient(image)
    gradXY = np.hypot(gradX, gradY)
    
    mask_fz = gradXY > tol_grad

    fz_grid = np.zeros(mask_fz.shape)
    fz_grid[mask_fz] = 1

    if iter_dilation:
        fz_grid = ndimage.binary_dilation(fz_grid, iterations=iter_dilation)
    
    if mask:
        fz_grid[raster.data.mask] = np.nan
        
    fz_raster = gplately.Raster(data=fz_grid, extent='global')    
    
    if return_gradient:
        return fz_raster, gradXY
    else:
        return fz_raster


def reconstruct_fracture_zones(time, return_grid=False):
    subduction_data = model.tessellate_subduction_zones(time, np.deg2rad(0.2), ignore_warnings=True)
    trench_lons = subduction_data[:,0]
    trench_lats = subduction_data[:,1]
    trench_norm = subduction_data[:,7]

    # store these for later
    subduction_lons = trench_lons.copy()
    subduction_lats = trench_lats.copy()
    
    dlon = -2.5*np.sin(np.radians(trench_norm))
    dlat = -2.5*np.cos(np.radians(trench_norm))
    
    trench_lons += dlon
    trench_lats += dlat
    
    agegrid_raster = gplately.Raster(filename=agegrid_filename.format(time))
    fz_raster, gradXY = grad(agegrid_raster, tol_grad=2, iter_dilation=0, mask=False, return_gradient=True)
    mask_raster = fz_raster.data >= 1
    fz_raster.data[mask_raster] = gradXY[mask_raster]
    
    trench_fz = fz_raster.interpolate(trench_lons, trench_lats, method='nearest')
    
    # mask points where fracture zone intersects a subduction zone
    mask_trench_fz = trench_fz > 0
    trench_fz   = trench_fz[mask_trench_fz]
    trench_lons = subduction_lons[mask_trench_fz]
    trench_lats = subduction_lats[mask_trench_fz]
    
    if return_grid:
        return trench_lons, trench_lats, trench_fz, fz_raster.data
    else:
        return trench_lons, trench_lats, trench_fz

# In[22]:


def plot_fz_timseries(time):
    fz_lons, fz_lats, fz_mag, fz_raster = reconstruct_fracture_zones(time, return_grid=True)

    LIP_features = model.reconstruct(LIP_filename, time)
    
    # set up map plot
    fig = plt.figure(figsize=(7.5,7.5))
    ax = fig.add_subplot(111, projection=ccrs.Orthographic(70, 0))
    ax.set_global()
    ax.gridlines(color='0.7', linestyle=':', xlocs=np.arange(-180,180,30), ylocs=np.arange(-90,90,30))


    # Plot shapefile features, subduction zones and MOR boundaries at 50 Ma
    gplot.time = time # Ma
    # gplot.plot_continent_ocean_boundaries(ax, color='b', alpha=0.05)
    gplot.plot_grid(ax, fz_raster, origin='lower', cmap='RdPu', vmin=0, vmax=2)
    gplot.plot_continents(ax, facecolor='0.8')
    gplot.plot_coastlines(ax, color='0.5')
    gplot.plot_ridges_and_transforms(ax, color='red', zorder=9)
    gplot.plot_feature(ax, LIP_features, color='DarkRed', alpha=0.25, zorder=9)

    ax.scatter(fz_lons, fz_lats, c='yellow', transform=ccrs.PlateCarree())

    gplot.plot_trenches(ax, color='k', zorder=9)
    gplot.plot_subduction_teeth(ax, color='k', zorder=10)
    
    
    # plot metals
    legend_elements = []
    for i, sheet in enumerate(sheets):
        gpts = pts_dict[sheet]
        df = metal_dict[sheet]
        
        ages = df['Age (Ga)']*1000
        mask_ages = ages >= time
        
        # create label for each sheet and add commodities
        label = ""
        size  = np.zeros(df.shape[0])
        for commodity in commodities:
            if commodity in df:
                label += "{} + ".format(commodity[:-5])
                size  += df[commodity].fillna(0.0).to_numpy()
        label = label[:-3] + " ({})".format(sheet)
        
        if mask_ages.any():
            # reconstruct
            rlons, rlats = gpts.reconstruct(time, return_array=True)
            
            # print(len(size), np.count_nonzero(mask_ages), df.shape)

            sc = ax.scatter(rlons[mask_ages], rlats[mask_ages], s=10+size[mask_ages]*2, 
                            marker=symbols[i], cmap='YlGnBu',
                            color='C{}'.format(i),
                            label=label, transform=ccrs.PlateCarree(),
                            edgecolor='k', linewidth=0.5, zorder=11)
            
        # create legend elements
        legend_elements.append( Line2D([0],[0], color='C{}'.format(i), marker=symbols[i], label=label,
                                       linestyle='none', markeredgecolor='k', markeredgewidth=0.5) )
    
    fig.legend(handles=legend_elements, loc='center', frameon=False, bbox_to_anchor=(0.5,0), ncol=2,
               title='Mineral deposit types', title_fontsize=12)

    fig.text(0.17, 0.8, "{:4d} Ma".format(time), fontsize=12)
    
    fig.savefig("snapshots/fz_metals_{:04d}Ma.png".format(time), dpi=300, bbox_inches='tight')
    plt.close(fig)
    return None


# In[9]:


from joblib import Parallel, delayed


if __name__ == "__main__":
    _ = Parallel(n_jobs=-3, backend='multiprocessing', verbose=1)(delayed(plot_fz_timseries) (time,) for time in reconstruction_times)


# Create a video using this command:
# 
# ```sh
# cat $(ls -r snapshots/slab_depth_*) | ffmpeg -y -f image2pipe -r 8 -i - -c:v h264_videotoolbox -b:v 6000k slab_dip_clennett_v4_no_plateaus.mp4
# ```

# In[ ]:




