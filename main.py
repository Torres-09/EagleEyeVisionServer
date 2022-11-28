import datetime
import os

from fastapi import FastAPI, File, UploadFile, Path, Query
import MyAwsS3
from pydantic import BaseModel
import logging
from typing import Union

import desmoke

app = FastAPI()

bucket_name = 'eagleeyes'
resultVideo_url = "/home/ubuntu/result/result.mp4"


class Item(BaseModel):
    filename: str


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/fastapi-test")
def test_api():
    return "hi"


@app.post("/fastapi-test2")
def test_api(example: str):
    return example


@app.post("/upload-file")
def upload_file(item: Item):
    MyAwsS3.upload_file(item.filename, bucket_name, "listen")
    return "okay"


@app.post("/upload-fileV3")
def upload_file(
        filename: str = Path,
        filename1: str = Path,
        file: str = Path,
        file1: str = Path
):
    # ( 연기를 제거하려는 원본영상의 경로, 최종 결과물이 저장되는 경로, 파일 이름(확장자포함) )
    # 원본영상은 java 서버에서 제거 , 영상처리 후 결과물은 여기서 제거해야 함.
    # desmoke.desmoke_video(filename, resultVideo_url, "result.mp4")
    MyAwsS3.upload_file(resultVideo_url, bucket_name, file)

    if os.path.exists(resultVideo_url)
        os.remove(resultVideo_url)

    MyAwsS3.upload_file(filename1, bucket_name, file1)
    return "okay"