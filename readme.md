Welcome to the `readme`. The following steps are used to create the dataset for the project: **Deep Neural Network Regression for Digital Surface Model Generation with Sentinel-2 Imagery**

- Step 1: Placing datasets\
To setup the system, you need to download to datasets. The North Rhine-Westphalia 
  nDsm is taken from [here](https://www.opengeodata.nrw.de/produkte/geobasis/hm/ndom50_tiff/ndom50_tiff/)
  and sentinel data from [here](https://download.geoservice.dlr.de/S2_L3A_WASP/files/32/U/). From latter one,
  download the subfolders LB,LC,MB and MC. Place them both in empty directories apart from each other. For example
  your structure could look like:
  \
  \
  ~/project_name/\
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|- ndsm (empty dir cearted by you)\
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|- ndom50_32280_5652_1_nw_2019.tif\
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|- ...\
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|- sentinel (empty dir cearted by you)\
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|- LB\
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|- 2020\
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|- 2021\
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|- SENTINEL2A_20150715-000000-000_L3A_T32ULB_C_V1-2\
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|- ...\
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|- LC\
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|- ...
  \
  For the nDsm dataset, a download script is provided in `dataset → download.py`. You can modify the output path of this script in `dataset → helper → helper.py` by the variable `dom_path`. 
  \
  &nbsp;
- Step 2: Inserting a positiontest\
The matching step later on needs to match the position of each nDsm tile to a sentinel
  subfolder (LB, LC, MB, MC), as they describe different areas. For this purpose we create a folder
  in every MB, MC, LB, LC subfolder called `positiontest`. We place a random sentinel tif of the current subfoler (LB,LC,MB,MC) in it (copied) to always be able to compare the nDSM tile position to the sentinel subfolder one without accessing file deeper in the sentinel structure.
  \
  \
  ~/project_name/\
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|- sentinel\
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|- LB\
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|- positiontest\
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|- SENTINEL2A_20150715-000000-000_L3A_T32ULB_C_V1-2_B7.tif\
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;| ...
  \
  &nbsp;
  
- Step 3: Executing scripts\
There are a couple of scripts needed to manipulate the dataset to get it ready
  for the neural network which takes the data in. Please execute every script from the base directory where your `dataset` and `network` folder is located. At first configure the `dataset_helper.py` in 
  `dataset → helper → helper.py`. The variables descriptions are commented in the file itself.
 Next, execute the walkdom script in `dataset → walkdom.py`. It's going to cut the ndsm tiles into smaller
 pieces and transforms the coordinate system to the one that sentinel data uses. After that, execute the `matching.py` script in 
 `dataset → matching.py`. It performs a time- and position matching for each ndsm tile with its corresponding sentinel part.
\
For error correction pls execute `dataset → correction → resolute_detection.py` at first. It will detect false data frames and save it to a list in `network → outliers_unchecked.txt`. Afterwards to complete the correction routine, run `dataset → correction → rule_checker.py`. It will go through every data frame, the detection has just marked and only really sort out data frames with obvious errors by creating yet another list in `network → outliers_checked_stayed.txt`. The rest will be shown to you. You can manually sort out the frame with just hitting enter in the console, or write down an arbitary letter and hit enter to keep your frame in the dataset. The kept ones will be shown in `network → outliers_checked_sorted_out.txt.`
 For each ndsm tile it produces an `.npz` data_frame containing the sentinel and the ndsm tiles. You can specifically look at a tile with the `dataset → helper → dataframe_viewer.py` script, which will plot you the nDSM tile with its corresponding sentinel crops. Keep in mind that `walkdom.py` and `matching.py`  depend on `dataset_helper.py`. As now the dataset is clean and ready for training, we move them into their training directories via executing `dataset → mover → move_split.py`.

- Step 4: Running the network\
The network is configured via the `network → network_helper.py` script. Also here all variables are commented out with an explanation and those variables will act as dependency paths for all network related scripts. To finally run the process, execute `network → training.py`. When training is done up to an epoch that satifies your requirements or when the earlystopping is reached, you may execute `network → result_script.py` to optain the results from the test set. 
