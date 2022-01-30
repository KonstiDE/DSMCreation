import json
import urllib.request as ulib
import os

from helper.dataset_helper import (
    dom_path
)

BASE_DOWNLOAD_URL = "https://www.opengeodata.nrw.de/produkte/geobasis/hm/ndom50_tiff/ndom50_tiff"
BASE_SAVE_PATH = dom_path


def download():
    with ulib.urlopen("https://www.opengeodata.nrw.de/produkte/geobasis/hm/ndom50_tiff/ndom50_tiff/index.json") as url:
        data = json.loads(url.read().decode())

        datafiles = data['datasets'][0]['files']

        for i in range(0, len(datafiles)):
            current = datafiles[i]
            current_fn = current['name']

            if current_fn not in os.listdir(BASE_SAVE_PATH):
                ulib.urlretrieve(
                    "{}/{}".format(BASE_DOWNLOAD_URL, current_fn),
                    "{}/{}".format(BASE_SAVE_PATH, current_fn)
                )
            else:
                print(current_fn + " is already downloaded")


if __name__ == '__main__':
    download()
