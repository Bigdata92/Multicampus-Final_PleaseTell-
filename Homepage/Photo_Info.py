from PIL import Image
from PIL.ExifTags import TAGS
import numpy as np
import pandas as pd
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
    date = '. '.join(ymd.split(':'))
    ymd_result = ''.join(ymd.split(':'))

    return date, ymd_result

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
    sido1 = response['results'][0]['region']['area1']['name']
    sido2 = response['results'][0]['region']['area2']['name']
    sido = sido1 + ' ' + sido2

    return sido, sido1, sido2

def photo_weather(key_filepath_w, sido, ymd_result):
    location_code = 'C:/Workspace/python/빅데이터 지능형서비스 개발 팀프로젝트/Final Project/Data/areacode.csv'
    ac = pd.read_csv(location_code)
    with open(key_filepath_w) as f:
        api_key = f.read()
    for location in ac.지점명:
        if location in sido.split()[1]:
            stnIds = list(ac[ac.지점명 == location].지점)[0]
            break
        elif location in sido.split()[0]:
            stnIds = list(ac[ac.지점명 == location].지점)[0]
            break
        else:
            stnIds = '일치하는 값이 없습니다.'
    startDt = ymd_result
    url = 'http://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList'
    params ={'serviceKey' : api_key, 'pageNo' : '1', 'numOfRows' : '10', 'dataType' : 'json', 'dataCd' : 'ASOS', \
             'dateCd' : 'DAY', 'startDt' : startDt, 'endDt' : startDt, 'stnIds' : stnIds }
    # sumRn : 일 강수량 / maxTa : 최고기온 / minTa : 최저기온 / avgTa 평균온도 / avgRhm : 평균 상대습도
    response = requests.get(url, params=params).json()
    maxTa = response['response']['body']['items']['item'][0]['maxTa']
    avgTa = response['response']['body']['items']['item'][0]['avgTa']
    minTa = response['response']['body']['items']['item'][0]['minTa']
    sumRn = response['response']['body']['items']['item'][0]['sumRn']
    avgRhm = response['response']['body']['items']['item'][0]['avgRhm']

    return maxTa, avgTa, minTa, sumRn, avgRhm

