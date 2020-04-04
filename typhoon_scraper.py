import requests
from bs4 import BeautifulSoup
import csv
import os
from io import StringIO
from PIL import Image
"""
    Script for scraping typhoon data from Digital Typhoon (http://agora.ex.nii.ac.jp/~kitamoto/)\n
    All scraped data shall only used for this project purpose and data will be deleted after this project.\n
    Credit to Prof KITAMOTO and the National Institute of Informatics. 誠にありがとうございます。\n
 """

# Config
SAVE_LOCATION = 'dataset'  # Make sure you opened this folder!
# For NW pacific
TYPHOONS_BY_YEAR_URL = "http://agora.ex.nii.ac.jp/digital-typhoon/year/wnp/{}.html.en"
TYPHOON_TRACK_URL = 'http://agora.ex.nii.ac.jp/digital-typhoon/summary/wnp/l/{}.html.en'
TYPHOON_IMAGE_URL_PREFIX = 'http://agora.ex.nii.ac.jp'
START_YEAR = 2009
END_YEAR = 2020
CSV_HEADER = ['id', 'name', 'year', 'month', 'day', 'hour', 'lat',
              'long', 'pressure', 'wind', 'class', 'img', 'chart']

typhoon_ids_by_year = {}
for yr in range(START_YEAR, END_YEAR, 1):
    print('Getting typhoon ids from year {}'.format(yr))
    url = TYPHOONS_BY_YEAR_URL.format(yr)
    print('URL:{}'.format(url))
    resp = requests.get(url)
    resp.encoding = 'utf-8'
    soup = BeautifulSoup(resp.text, 'lxml')
    typhoon_ids = [a.text for a in soup.find_all(
        'a') if '/digital-typhoon/summary/wnp/s/' in a['href']]
    typhoon_ids_by_year[yr] = typhoon_ids


def fetchBDImage(url, fname, path):
    print("Downloading BD Image for {}".format(fname))
    resp = requests.get(url)
    resp.encoding = 'utf-8'
    soup = BeautifulSoup(resp.text, 'lxml')
    href = soup.find('a', text='Magnify this',
                     href=lambda x: x and '/bd/' in x)['href']
    img = Image.open(requests.get(
        f"{TYPHOON_IMAGE_URL_PREFIX}{href}", stream=True).raw)
    img.save(f"{path}/{fname}.jpg")
    return fname


for yr, ids in typhoon_ids_by_year.items():
    print('Getting typhoon data from year {}'.format(yr))
    directory = f"{SAVE_LOCATION}/{yr}"
    for id in ids:
        url = TYPHOON_TRACK_URL.format(id)
        resp = requests.get(url)
        resp.encoding = 'utf-8'
        soup = BeautifulSoup(resp.text, 'lxml')
        raw_track_datas = soup.find_all(
            'tr', {"class": lambda c: c and 'ROW' in c})
        path = f"{directory}/{id}"
        name_raw = soup.find('div', {'class': 'TYNAME'}).text
        name = name_raw[name_raw.find('(')+1:name_raw.find(')')]
        try:
            if not os.path.exists(path):
                os.makedirs(path)
        except OSError:
            print('Error: Creating directory. ' + directory)
        print('Getting typhoon {} {} data'.format(id, name))
        with open(f'./{path}/{id}{name}.csv', mode='w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_HEADER)
            writer.writeheader()
            for day in raw_track_datas:
                td = day.find_all('td')
                data = {'name': name, 'id': id}
                for idx, d in enumerate(td):
                    if d.find('a'):
                        data[CSV_HEADER[idx+2]] = TYPHOON_IMAGE_URL_PREFIX + \
                            d.find('a')['href']
                    else:
                        data[CSV_HEADER[idx+2]] = d.text
                fname = f"{id}-{data['year']}{data['month'].zfill(2)}{data['day'].zfill(2)}-{data['hour'].zfill(2)}"
                data['img'] = fetchBDImage(data['img'], fname, path)
                writer.writerow(data)
