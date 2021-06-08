# this is a Ubuntu bash script to prepare the CMEMS data
# To run it navigate to the folder where all CMEMS phy_030... nc-files are stroed. In my case:
# cd /mnt/h/Eigene\ Dateien/Studium/10.\ Semester/NIOZ/data/CMEMS/

#  create a subfolder to store the regridded nc files in
mkdir regridded

# go to this new directory
cd regridded/

# get all filenames in parent directory
allfnames=`ls ../*.nc`

# Use cdo to extract variables uo, vo and mlotst and regrid the nc file to a 0.5x0.5° resolution
for eachfile in $allfnames
	do
	   cdo -L remapbil,r720x360 -select,name=uo,vo,mlotst $eachfile ${eachfile:3:38}
	done

# merge all the regridded nc files and create a single nc file with a time dimension
cdo mergetime *.nc CMEMS_phy_030_uo_vo_mlotst_1993_2018.nc

# remove the seasonality from the data
cdo -L -ymonsub CMEMS_phy_030_uo_vo_mlotst_1993_2018.nc -ymonmean CMEMS_phy_030_uo_vo_mlotst_1993_2018.nc CMEMS_phy_030_uo_vo_mlotst_1993_2018_deseasoned.nc