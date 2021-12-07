# 빅데이터 지능형서비스 개발과정 팀프로젝트(Final)

## 멀티캠퍼스, 서울

### 기간 : 21. 10. 29(금) ~ 12. 9(목)

**세부일정** 

| 단계 |               활동               |       수행기간       |
| :--: | :------------------------------: | :------------------: |
| 기획 | 주제 정하기 / 아이디어 자료 찾기 | 10/29(금) ~ 11/5(금) |
| 분석 |        EDA / 사진정보추출        | 11/8(월) ~ 11/12(금) |
| 학습 |    YOLOv5 / Image Captioning     | 11/8(월) ~ 11/19(금) |
| 구현 |  GPT / Google Vision / FAST API  | 11/22(월) ~ 12/3(금) |
| 종료 |      PPT / 발표준비 / 발표       | 12/6(월) ~ 12/8(목)  |

**팀이름 : 플리즈 텔!**

**프로젝트 역할**(공동 작업파트 : 주제선정 / 데이터 수집(크롤링, 자료조사 등) / 데이터 전처리)

| 팀원   |                             역할                             |
| ------ | :----------------------------------------------------------: |
| 김나영 |            EDA / 이미지캡셔닝 모델작성 / PPT작성             |
| 서동규 |    사진정보추출 / GPT모델 작성 / 객체탐지모델 작성 / 발표    |
| 조정범 | GPT모델 작성 / 이미지 캡셔닝 모델 작성 / BLEU 코드 작성 / FASTAPI |
| 송지섭 |          객체탐지모델 작성 / 데이터 가공 / PPT작성           |
| 전봉수 |              EDA / 객체탐지모델 작성 / PPT작성               |
| 박준민 |       사진정보추출 / 객체탐지모델 작성 / CFR / FASTAPI       |

**주제**

- 사진을 올리면 여러가지 테마로 글을 작성해주는 작문 서비스


 **데이터 설명**

- 크롤링 데이터(Melon, 브런치, 비짓제주, 위시빈, 문학광장 etc)
- 구글을 활용한 자체 설문조사
- MSCOCO 2014 train, val(이미지) / AI Hub(MSCOCO 한국어 캡셔닝)
- AI Hub 멀티모달 데이터(이미지 / 캡셔닝)

**기대 효과**

- 작문에 어려움을 겪는 사람들에게 초고를 제시해줌으로써 작문의 어려움 완화
- 생각하지못한 관점으로 쓰여진 글을 제시해줌으로써 즐거움 제공

**분석 내용**

- **Selenium / Beautifulsoup**를 활용한 **데이터 크롤링**
- **파이썬 패키지**를(re, pandas등) 활용한 **데이터 전처리**
- **Google Colab**를 활용한 **이미지 캡셔닝 모델** 훈련 및 작성 / **KoGPT2 Finetuning**

**사용 모델 및 API**

- **기상청 API(날씨 정보), NAVER API(위치정보)**
- **객체탐지모델**(YOLOv5, Detectron2, **GoogleVisionAI**)
- **이미지캡셔닝모델(Google Tensorflow 코드 변형)**
  - https://www.tensorflow.org/tutorials/text/image_captioning?hl=ko
- **GPT모델(KoGPT2)**
- **FASTAPI를 활용한 테스트 서비스 작성**(HTML : HTML5 템플릿 변형 / CSS: 코드펜 및 구글링 코드활용)

**참고자료 및 사이트 출처**

1. 오픈서베이 콘텐츠 트렌드 리포트 2021 / [https://blog.opensurvey.co.kr/trendreport/contents-2021](https://blog.opensurvey.co.kr/trendreport/contents-2021/)
2. 구글 트렌드 검색  / https://trends.google.co.kr/trends/?geo=KR
3. 네이버 트렌드 검색 / [https://datalab.naver.com/keyword](https://datalab.naver.com/keyword/)
4. 박대아, 대학생의 글쓰기 인식에 대한 추적조사연구 / https://papersearch.net/thesis/article.asp?key=3814855
5. 네이버 클로바 API / https://clova.ai/ko/aisolutions/
6. 공공데이터포털(기상청 API) /  https://www.data.go.kr/index.do
7. Exifread / https://pypi.org/project/ExifRead/
8. Detectron2 / https://ai.facebook.com/tools/detectron2/
9. YOLOv5 / https://github.com/ultralytics/yolov5
10. roboflow / https://roboflow.com/
11. GoogleVisionAI / https://cloud.google.com/vision
12. 이미지캡셔닝 / https://www.tensorflow.org/tutorials/text/image_captioning?hl=ko
13. 텐서플로우 이미지캡셔닝 / https://www.tensorflow.org/tutorials/text/image_captioning?hl=ko
14. AI HUB 멀티모달 / https://aihub.or.kr/aidata/135
15. KoGPT2 / https://github.com/SKT-AI/KoGPT2
16. 멜론 / https://www.melon.com/
17. 브런치 / https://brunch.co.kr/
18. 대한민국 구석구석 / https://korean.visitkorea.or.kr/main/main.do#home
19. 비짓제주 / https://www.visitjeju.net/kr
20. FASTAPI / https://fastapi.tiangolo.com/

**※ 관련코드들은 최종코드에서 확인하시고, 홈페이지 및 테스트 서비스 관련 파일들은 Homepage폴더를 참조하시기 바랍니다.**





