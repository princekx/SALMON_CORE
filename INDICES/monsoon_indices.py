import xarray as xr
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import datetime
from datetime import timedelta

ct=datetime.datetime.now().date()
print(ct)

def subtract_days_from_date(date,days):
    subtracted_date=pd.to_datetime(date)-timedelta(days=days)
    subtracted_date=subtracted_date.strftime("%d-%m-%Y")

    return subtracted_date

adate=subtract_days_from_date(ct,1)

from datetime import datetime
adate1=datetime.strptime(adate, "%d-%m-%Y")
yyyy = adate1.strftime("%Y")
mm = adate1.strftime("%m")
dd = adate1.strftime("%d")

dir = "/home/index/GEFS/"
url = f'http://nomads.ncep.noaa.gov:80/dods/gefs/gefs{yyyy}{mm}{dd}/gefs_pgrb2ap5_all_00z'

ds = xr.open_dataset(url, engine='netcdf4')
dsu=ds['ugrdprs']
dsv=ds['vgrdprs']


#added 9 December 2024
#slice the data into smaller region to better handle the data
lat_box=[0,30]
lon_box=[90,160]

dsu_region=dsu.sel(lon=slice(*lon_box),lat=slice(*lat_box))
dsv_region=dsv.sel(lon=slice(*lon_box),lat=slice(*lat_box))

dsu850=dsu_region.sel(lev=850).mean(dim='ens')
dsu925=dsu_region.sel(lev=925).mean(dim='ens')
dsv925=dsv_region.sel(lev=925).mean(dim='ens')

u850=dsu850.resample(time='D').mean()
u925=dsu925.resample(time='D').mean()
v925=dsv925.resample(time='D').mean()

u850

#SWMI (Southwest Monsoon Index)
box1=u850.sel(lon=slice(90,130),lat=slice(5,15)).mean(dim=['lon','lat'])
box2=u850.sel(lon=slice(100.75,103.25),lat=slice(1.75,4.25)).mean(dim=['lon','lat'])
swmi1=box2-box1

df_swmi1=swmi1.to_dataframe().reset_index()
df_swmi1.to_csv(f'{dir}swmi1_'+str(adate)+'.csv',index=False)

sns.lineplot(data=df_swmi1,x='time',y='ugrdprs',marker='p')
plt.title('GEFS SWMI Forecast Updated:'+str(ct),fontsize=10)
plt.xlabel('Date',fontsize=8)
plt.ylabel('SWMI1 (m/s)',fontsize=8)
plt.xticks(rotation=90)
plt.tick_params(labelsize=8)
plt.axhline(y=0.0, color='red', linestyle='--', linewidth=2)
plt.tight_layout()
plt.savefig(f'{dir}gfs_shear.gfsgefs.png',bbox_inches='tight',dpi=300)
plt.close('all')
#Sending data to 134
os.system(f'scp -rv {dir}gfs_shear.gfsgefs.png KC@10.41.16.134:/var/www/html/ideas/southwest_monsoon/')

################################################
#SWMI2 (Southwest Monsoon Index 2)
swmi2=u850.sel(lon=slice(100,115),lat=slice(5,10)).mean(dim=['lon','lat'])

df_swmi2=swmi2.to_dataframe().reset_index()
df_swmi2.to_csv(f'{dir}swmi2_'+str(adate)+'.csv',index=False)

sns.lineplot(data=df_swmi2,x='time',y='ugrdprs',marker='p')
plt.title('GEFS SWMI2 Forecast Updated:'+str(ct),fontsize=10)
plt.xlabel('Date',fontsize=8)
plt.ylabel('SWMI2 (m/s)',fontsize=8)
plt.xticks(rotation=90)
plt.tick_params(labelsize=8)
plt.axhline(y=0.0, color='red', linestyle='--', linewidth=2)
plt.tight_layout()
plt.savefig(f'{dir}GEFS_SWMI2.png',bbox_inches='tight',dpi=300)
plt.close('all')
os.system(f'scp -rv {dir}GEFS_SWMI2.png KC@10.41.16.134:/var/www/html/ideas/southwest_monsoon/')


#############################################
#NEMI (Northeast Monsoon Index)
n850=u850.sel(lon=slice(102.5,105),lat=slice(3.75,6.25)).mean(dim=['lon','lat'])
n925=u925.sel(lon=slice(102.5,105),lat=slice(3.75,6.25)).mean(dim=['lon','lat'])

df850=n850.to_dataframe().reset_index()
df925=n925.to_dataframe().reset_index()
df925=df925.rename(columns={'ugrdprs':'u925'})
df850=df850.rename(columns={'ugrdprs':'u850'})

df_nemi=pd.merge(df850,df925,on='time')
df_nemi['average']=df_nemi[['u850','u925']].mean(axis=1)
df_nemi.to_csv(f'{dir}nemi_'+str(adate)+'.csv',index=False)

sns.lineplot(data=df_nemi,x='time',y='average',marker='p')
plt.title('GEFS NEMI Forecast Updated:'+str(ct),fontsize=10)
plt.xlabel('Date',fontsize=8)
plt.ylabel('NEMI (m/s)',fontsize=8)
plt.xticks(rotation=90)
plt.tick_params(labelsize=8)
plt.axhline(y=-2.5, color='red', linestyle='--', linewidth=2)
plt.tight_layout()
plt.savefig(f'{dir}gfs_onset.gfsgefs.png',bbox_inches='tight',dpi=300)
plt.close('all')
os.system(f'scp -rv {dir}gfs_onset.gfsgefs.png KC@10.41.16.134:/var/www/html/ideas/northeast_monsoon/')

#############################################
#NEMO (Northeast Monsoon Onset)
nemo=v925.sel(lon=slice(107,115),lat=slice(5,15)).mean(dim=['lon','lat'])

df_nemo=nemo.to_dataframe().reset_index()

df_nemo.to_csv(f'{dir}nemo_'+str(adate)+'.csv',index=False)

sns.lineplot(data=df_nemo,x='time',y='vgrdprs',marker='p')
plt.title('GEFS NEMO Forecast Updated:'+str(ct),fontsize=10)
plt.xlabel('Date',fontsize=8)
plt.ylabel('NEMO (m/s)',fontsize=8)
plt.xticks(rotation=90)
plt.tick_params(labelsize=8)
#plt.axhline(y=0.0, color='red', linestyle='--', linewidth=2)
plt.tight_layout()
plt.savefig(f'{dir}GEFS_NEMO',bbox_inches='tight',dpi=300)
plt.close('all')
os.system(f'scp -rv {dir}GEFS_NEMO.png KC@10.41.16.134:/var/www/html/ideas/northeast_monsoon/')

###########################################
#Easterly Surge
esi=u925.sel(lon=120,lat=slice(7.5,15)).mean(dim='lat')
df_esi=esi.to_dataframe().reset_index()

#Meridional Surge
msi=v925.sel(lon=slice(110,117.5),lat=15).mean(dim='lon')
df_msi=msi.to_dataframe().reset_index()

sns.lineplot(data=df_esi,x='time',y='ugrdprs',marker='p',label='ESI')
sns.lineplot(data=df_msi,x='time',y='vgrdprs',marker='s',label='MSI')
plt.title('GEFS MESI Forecast Analysis Time :(00Z '+str(adate)+')',fontsize=10)
plt.xlabel('Date',fontsize=8)
plt.ylabel('u/v-wind (m/s)',fontsize=8)
plt.xticks(rotation=90)
plt.axhline(y=-8, color='red', linestyle='--',linewidth=2)
plt.tight_layout()
plt.savefig(f'{dir}gfs_mesi.gfsgefs.png',bbox_inches='tight',dpi=300)
plt.close('all')
os.system(f'scp -rv {dir}gfs_mesi.gfsgefs.png KC@10.41.16.134:/var/www/html/ideas/northeast_monsoon/')

