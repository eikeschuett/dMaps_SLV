# This is a Ubuntu bash script that downloads and prepares the AVISO monthly mean sea level anomalies dataset.
# first navigate to a directory where you want to store all raw nc files. In my case e.g. 
# cd /mnt/h/Eigene\ Dateien/Studium/10.\ Semester/NIOZ/data/AVISO/raw/

mkdir AVISO/
cd AVISO/
# Then download the data using wget. You need to register at AVISO to download the data and insert your username and password in the next line.
wget -c -r -np -nH -A nc.gz --user <username> --password <password> --cut-dirs=1 -erobots=off --no-check-certificate ftp://ftp-access.aviso.altimetry.fr/climatology/global/delayed-time/monthly_mean/

# The downloaded files are zipped. Let's unzip them.
# get all filenames in directory
allfnames=`ls *.gz`

# Unzip each file and delete the zipped version
for eachfile in $allfnames
	do
	   gzip -d $eachfile
	done

# Concatenate all individual nc-files into a new one which contains a time variable and the data of each month
cdo mergetime *.nc AVISO_MSLA_1993-2020.nc

# Remove the seasonal cycle
cdo -L -ymonsub AVISO_MSLA_1993-2020.nc -ymonmean AVISO_MSLA_1993-2020.nc tmp.nc

# Remove linear trend
cdo detrend tmp.nc AVISO_MSLA_1993-2020_prep_2_deg.nc

# remove temp file
rm tmp.nc

# Next, use the Python code to apply the Gausssian filter and run deltaMaps on the dataset