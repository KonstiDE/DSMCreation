Welcome to the `readme`. The following steps are used to create the dataset for the project: **Deep Neural Network Regression for Digital Surface Model Generation with Sentinel-2 Imagery**

- Step 1: Placing datasets\
To setup the system, you to download to datasets. The North Rhine-Westphalia 
  nDsm from https://www.opengeodata.nrw.de/produkte/geobasis/hm/ndom50_tiff/ndom50_tiff/ 
  and sentinel data from https://download.geoservice.dlr.de/S2_L3A_WASP/files/32/U/, latter one
  the subfolders LB,LC,MB and MC. Place them both in empty directories apart from each other. For example
  your structure could look like:
  \
  \
  C:/Users/Administrator/project_data/\
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|- ndsm\
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|- ndom50_32280_5652_1_nw_2019.tif\
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|- ...\
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|- sentinel\
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|- LB\
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|- 2020\
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|- 2021\
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|- SENTINEL2A_20150715-000000-000_L3A_T32ULB_C_V1-2\
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|- ...\
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|- LC\
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|- ...
  \
  For the nDsm dataset, a download script is provided in `NRW → dataset → download.py`. You can modify the output path of this script in `NRW → dataset → helper → helper.py` by the variable `dom_path`. 
  \
  \
  \
  &nbsp;
- Step 2: Inserting a positiontest\
The matching step later on needs to match the position of each nDsm tile to a sentinel
  subfolder (LB, LC, MB, MC), as they describe different areas. For this purpose we create a folder
  in every MB, MC, LB, LC subfolder called `positiontest`. We place a random sentinel tif of the current subfoler (LB,LC,MB,MC) in it.
  \
  \
  C:/Users/Administrator/project_data/\
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|- sentinel\
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|- LB\
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|- positiontest\
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|- SENTINEL2A_20150715-000000-000_L3A_T32ULB_C_V1-2_B7.tif\
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;| ...
  \
  \
  \
  &nbsp;
  
- Step 3: Executing scripts\
There are a couple of scripts needed to manipulate the dataset to get it ready
  for the neural network which takes the data on. At first configure the `helper.py` in 
  `NRW → dataset → helper → helper.py`. The variables descriptions are commented on the `helper.py` file.
 Next, execute the walkdom script in `NRW → dataset → walkdom.py`. It's going to cut the ndsm tiles into smaller
 pieces and transforms the coordinate system to the one that sentinel data uses. After that, execute the `matching.py` script in 
 `NRW → dataset → matching.py`. It performs a time- and position matching for each ndsm tile with its corresponding sentinel part.
 For each ndsm tile it produces an `.npz` data_frame containing the sentinel and the ndsm tiles.
