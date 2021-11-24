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
from PIL.Image import Image
import shutil
import json
import Captioning_Okt as cp
import KoGPT2_Branch as kgb
import KoGPT2_Wise as kgbw
import KoGPT2_essay as kgbe
import yolo5 as y5
from hanspell import spell_checker

app = FastAPI()

app.mount("/static", StaticFiles(directory = "C:/Workspace/python/빅데이터 지능형서비스 개발 팀프로젝트/Final Project/Homepage/static"), name = "static")
templates = Jinja2Templates(directory = "C:/Workspace/python/빅데이터 지능형서비스 개발 팀프로젝트/Final Project/Homepage/templates")

class Item(BaseModel):
    name: str

@app.get('/', response_class=HTMLResponse)
async def home(request : Request) :
    return templates.TemplateResponse("Service01(input).html", context={"request": request})

@app.post("/service01_test01/", status_code = 201)
async def testService01_out01(request : Request, img: UploadFile = File(...), gptModel: str = Form(...)):
    path = 'C:/Workspace/python/빅데이터 지능형서비스 개발 팀프로젝트/Final Project/Homepage/static/images/Upload_Images/'
    img_location = path + img.filename
    img_name = img.filename
    with open(img_location, "wb+") as file_object:
        file_object.write(img.file.read())
    img = {'image': open(img_location, 'rb')}

    for i in range(1, 6):
        caption, _ = cp.evaluate(img_location)
        caption = spell_checker.check(' '.join(caption[:-1]))
        globals()['caption_' + f'{i}'] = caption.checked

    object_list = list(set(y5.yolo(img_location)))
    object_list[0] = '# ' + object_list[0] 
    object_list = ' # '.join(object_list)

    gptModel = gptModel

    context = {'request': request, 'img_name' : img_name, 'caption_1' : caption_1, 'caption_2' : caption_2, \
                'caption_3' : caption_3, 'caption_4' : caption_4, 'caption_5' : caption_5, 'gptModel' : gptModel, \
                'object_list' : object_list}

    return templates.TemplateResponse("Service01(output01).html", context)

@app.post("/service01_test02/", status_code = 201)
async def testService01_out02(request : Request, img_name: str = Form(...), caption: str = Form(...), gptModel: str = Form(...), object_list: str = Form(...)):
    path = 'C:/Workspace/python/빅데이터 지능형서비스 개발 팀프로젝트/Final Project/Homepage/static/images/Upload_Images/'
    img_name = img_name
    img_location = path + img_name

    caption = caption
    gptModel = gptModel
    object_list = object_list

    sequence_list = []
    if gptModel == '수필':
        for _ in range(2):
            sequence = kgb.result_sequence(caption, 64)
            sequence_list.append(sequence)
        sequence = ' '.join(sequence_list)
    if gptModel == '명언':
        for _ in range(2):
            sequence = kgbw.result_sequence(caption, 64)
            sequence_list.append(sequence)
        sequence = ' '.join(sequence_list)
    if gptModel == '일기':
        for _ in range(2):
            sequence = kgb.result_sequence(caption, 64)
            sequence_list.append(sequence)
        sequence = ' '.join(sequence_list)

    context = {
        "request": request, 'caption' : caption, 'sequence' : sequence, 'img_name' : img_name, 'gptModel' : gptModel, \
        'object_list' : object_list
    }

    return templates.TemplateResponse("Service01(output02).html", context)

ngrok_tunnel = ngrok.connect(8000)
print ('Public URL:', ngrok_tunnel.public_url)
nest_asyncio.apply()
uvicorn.run(app, host='0.0.0.0', port=8000)