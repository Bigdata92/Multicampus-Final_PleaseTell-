from PIL import Image
from PIL.ExifTags import TAGS
import numpy as np
import requests

def photo_time(img_path):
    img = Image.open(img_path)
    img_info = img._getexif();
    img.close()
    taglabel = dict()

    for tag, value in img_info.items():
        decoded = TAGS.get(tag, tag)
        taglabel[decoded] = value

    date_raw = taglabel['DateTime']
    date_list = date_raw.split()
    ymd = date_list[0]
    date = '날짜 : ' + '. '.join(ymd.split(':'))
    return date

def photo_location(img_path, key_filepath):
    img = Image.open(img_path)
    img_info = img._getexif();
    img.close()
    taglabel = dict()

    for tag, value in img_info.items():
        decoded = TAGS.get(tag, tag)
        taglabel[decoded] = value
    
    latDeg = taglabel['GPSInfo'][2][0]
    latMin = taglabel['GPSInfo'][2][1]
    latSec = taglabel['GPSInfo'][2][2]
    # latDeg, latMin, latSec
    lonDeg = taglabel['GPSInfo'][4][0]
    lonMin = taglabel['GPSInfo'][4][1]
    lonSec = taglabel['GPSInfo'][4][2]
    # lonDeg, lonMin, lonSec
    x = (int(latDeg)+(int(latMin)/60)+(int(latSec)/3600))
    y = int(lonDeg)+(int(lonMin)/60)+(int(lonSec)/3600)
    Lat = f'''{str(int(latDeg))}°{str(int(latMin))}'{str(latSec)}"{taglabel["GPSInfo"][1]}'''
    Lon = str(int(lonDeg)) + "°" + (str(int(lonMin)) + "'") + (str(lonSec) + '"') + taglabel['GPSInfo'][3]

    # key 호출
    with open(key_filepath) as f:
        api_key = f.read()
    client_id = api_key.split('\n')[0]
    client_secret = api_key.split('\n')[1]


    # 네이버 api에 위치정보 요청(좌표값 활용)
    url = f"https://naveropenapi.apigw.ntruss.com/map-reversegeocode/v2/gc?coords={y},{x}&sourcecrs=epsg:4326&orders=legalcode&output=json&orders=addr,admcode,roadaddr"
    payload={}
    headers = {
    'X-NCP-APIGW-API-KEY-ID': client_id,
    'X-NCP-APIGW-API-KEY': client_secret
    }
    response = requests.get(url, headers=headers, data=payload).json()
    si = response['results'][0]['region']['area1']['name']
    do = response['results'][0]['region']['area2']['name']
    sido = si+ ' ' + do

    return sido, si, do
