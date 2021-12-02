from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.param_functions import File
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import RedirectResponse
from pydantic import BaseModel
from typing import List # 파일 여러 개 받기
from pyngrok import ngrok
import uvicorn
import nest_asyncio
from PIL import Image
from PIL.ExifTags import TAGS
import shutil
import json
import Captioning_Okt as cp
import Captioning_Okt as cp
import KoGPT2_명언 as kgbw
import KoGPT2_발라드 as kgbb
import KoGPT2_수필 as kgbe
import KoGPT2_시 as kgbp
import KoGPT2_여행 as kgbv
import KoGPT2_여행_비짓제주 as kgbj
import KoGPT2_자유형식_브런치 as kgbd
import KoGPT2_트로트 as kgbt
import yolo5 as y5
from hanspell import spell_checker
import googletrans 


app = FastAPI()

app.mount("/static", StaticFiles(directory = "C:/Workspace/python/빅데이터 지능형서비스 개발 팀프로젝트/Final Project/Homepage/static"), name = "static")
templates = Jinja2Templates(directory = "C:/Workspace/python/빅데이터 지능형서비스 개발 팀프로젝트/Final Project/Homepage/templates")

class Item(BaseModel):
    name: str

@app.get('/', response_class=HTMLResponse)
async def home(request : Request) :
    return templates.TemplateResponse("index.html", context={"request": request})

@app.get('/Service00/', response_class=HTMLResponse)
async def service00(request : Request) :
    return templates.TemplateResponse("Service00.html", context={"request": request})

@app.get('/service01_ct01/', response_class=HTMLResponse)
async def service_ct01(request : Request) :
    return templates.TemplateResponse("Service01_ct01(input).html", context={"request": request})

@app.get('/service01_ct02/', response_class=HTMLResponse)
async def service_ct02(request : Request) :
    return templates.TemplateResponse("Service01_ct02(input).html", context={"request": request})

@app.get('/service01_ct03/', response_class=HTMLResponse)
async def service_ct03(request : Request) :
    return templates.TemplateResponse("Service01_ct03(input).html", context={"request": request})

@app.get('/service01_ct04/', response_class=HTMLResponse)
async def service_ct04(request : Request) :
    return templates.TemplateResponse("Service01_ct04(input).html", context={"request": request})

@app.post("/service01_test01/", status_code = 201)
async def testService01_out01(request : Request, img: UploadFile = File(...), gptModel: str = Form(...)):
    path = 'C:/Workspace/python/빅데이터 지능형서비스 개발 팀프로젝트/Final Project/Homepage/static/images/Upload_Images/'
    img_location = path + img.filename
    img_name = img.filename
    with open(img_location, "wb+") as file_object:
        file_object.write(img.file.read())
    img = {'image': open(img_location, 'rb')}

    try:
        img = Image.open(img_location)
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
    except:
        date = '날짜 : 알 수없음'

    for i in range(1, 6):
        caption, _ = cp.evaluate(img_location)
        caption = spell_checker.check(' '.join(caption[:-1]))
        globals()['caption_' + f'{i}'] = caption.checked

    object_list = list(set(y5.yolo(img_location)))
    translator = googletrans.Translator()
    if len(object_list) != 0:   
        for i in range(len(object_list)): 
            object_list[i] = translator.translate(object_list[i], dest='ko').text
        object_list[0] = '# ' + object_list[0] 
        object_list = ' # '.join(object_list)
    else:
        object_list = '# 태그값 없음'

    gptModel = gptModel

    context = {'request': request, 'img_name' : img_name, 'caption_1' : caption_1, 'caption_2' : caption_2, \
                'caption_3' : caption_3, 'caption_4' : caption_4, 'caption_5' : caption_5, 'gptModel' : gptModel, \
                'object_list' : object_list, 'date' : date}

    return templates.TemplateResponse("Service01(output01).html", context)

@app.post("/service01_test02/", status_code = 201)
async def testService01_out02(request : Request, img_name: str = Form(...), caption: str = Form(...), gptModel: str = Form(...), \
                              object_list: str = Form(...), date: str = Form(...), sentence_length: int = Form(...)):
    path = 'C:/Workspace/python/빅데이터 지능형서비스 개발 팀프로젝트/Final Project/Homepage/static/images/Upload_Images/'
    img_name = img_name
    img_location = path + img_name

    caption = caption
    gptModel = gptModel
    object_list = object_list
    date = date

    try:
        sentence_length = sentence_length
    except:
        sentence_length = 2

    sequence_list = []
    if gptModel == '명언':
        for _ in range(sentence_length):
            sequence = kgbw.result_sequence(caption, 72)
            sequence_list.append(sequence)
        sequence = ' '.join(sequence_list)
    if gptModel == '발라드':
        for _ in range(sentence_length):
            sequence = kgbb.result_sequence(caption, 72)
            sequence_list.append(sequence)
        sequence = ' '.join(sequence_list)
    if gptModel == '비짓제주':
        for _ in range(sentence_length):
            sequence = kgbj.result_sequence(caption, 72)
            sequence_list.append(sequence)
        sequence = ' '.join(sequence_list)
    if gptModel == '트로트':
        for _ in range(sentence_length):
            sequence = kgbt.result_sequence(caption, 72)
            sequence_list.append(sequence)
        sequence = ' '.join(sequence_list)
    if gptModel == '시':
        for _ in range(sentence_length):
            sequence = kgbp.result_sequence(caption, 72)
            sequence_list.append(sequence)
        sequence = ' '.join(sequence_list)
    if gptModel == '여행':
        for _ in range(sentence_length):
            sequence = kgbv.result_sequence(caption, 72)
            sequence_list.append(sequence)
        sequence = ' '.join(sequence_list)
    if gptModel == '수필':
        for _ in range(sentence_length):
            sequence = kgbe.result_sequence(caption, 72)
            sequence_list.append(sequence)
        sequence = ' '.join(sequence_list)
    if gptModel == '일상일기':
        for _ in range(sentence_length):
            sequence = kgbd.result_sequence(caption, 72)
            sequence_list.append(sequence)
        sequence = ' '.join(sequence_list)

    context = {
        "request": request, 'caption' : caption, 'sequence' : sequence, 'img_name' : img_name, 'gptModel' : gptModel, \
        'object_list' : object_list, 'date' : date, 'sentence_length' : sentence_length
    }

    return templates.TemplateResponse("Service01(output02).html", context)

ngrok_tunnel = ngrok.connect(8000)
print ('Public URL:', ngrok_tunnel.public_url)
nest_asyncio.apply()
uvicorn.run(app, host='0.0.0.0', port=8000)