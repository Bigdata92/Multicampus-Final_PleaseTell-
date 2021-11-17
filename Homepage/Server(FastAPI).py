from fastapi import FastAPI, Form, Request
from pyngrok import ngrok
import uvicorn
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import nest_asyncio
from fastapi.staticfiles import StaticFiles


app = FastAPI()

app.mount("/static", StaticFiles(directory = "C:/Workspace/python/빅데이터 지능형서비스 개발 팀프로젝트/Final Project/Homepage/static"), name = "static")
templates = Jinja2Templates(directory = "C:/Workspace/python/빅데이터 지능형서비스 개발 팀프로젝트/Final Project/Homepage/templates")

@app.get('/', response_class=HTMLResponse)
async def home(request : Request) :
    return templates.TemplateResponse("Service01(input).html", context={"request": request})

# @app.get('/contact/')
# async def contact(request : Request) :
#     return templates.TemplateResponse("contact.html", context={"request": request})

# @app.get('/introduce_Service/', response_class=HTMLResponse)
# async def introduceService(request : Request) :
#     return templates.TemplateResponse("introduceService.html", context={"request": request})

# @app.get('/introduce_Service/test_map/', response_class=HTMLResponse)
# async def testMap(request : Request) :
#     return templates.TemplateResponse("test_map.html", context={"request": request})

# @app.post("/introduce_Service/testService_output/")
# async def testService(request : Request, content: str = Form(...)):
#     import pandas as pd
#     import numpy as np
#     from sklearn.metrics.pairwise import cosine_similarity
#     from sentence_transformers import SentenceTransformer, util
#     df = pd.read_csv('C:\\Workspace\\python\\빅데이터 지능형서비스 개발 팀프로젝트\\PJTJ4U\\조정범\\Data\\Dataset\\주소변환완료.csv')
#     embedding_jibun = np.load('C:\\Workspace\\python\\빅데이터 지능형서비스 개발 팀프로젝트\\PJTJ4U\\조정범\\Data\\Dataset\\임베딩자료(지번)\\embedding_result(지번).npy')
#     sentences = [juso for juso in df['지번주소']]
#     model = SentenceTransformer('all-mpnet-base-v2')
#     input = content
#     input_embeddings = model.encode(input)
#     input_embeddings = np.expand_dims(input_embeddings, axis = 0) 
#     cosine_scores = cosine_similarity(embedding_jibun, input_embeddings)
#     cosine_scores_list= cosine_scores.tolist()
#     result = f"'{input}'과 가장 유사한 주소 : "
#     answer = f"'{sentences[cosine_scores_list.index(max(cosine_scores_list))]}'"
#     return templates.TemplateResponse("testService(output).html", {"request":request, "answer" : answer, "result" : result})


ngrok_tunnel = ngrok.connect(8000)
print ('Public URL:', ngrok_tunnel.public_url)
nest_asyncio.apply()
uvicorn.run(app, host='0.0.0.0', port=8000)