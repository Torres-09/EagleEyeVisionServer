import datetime
import os

from fastapi import FastAPI, File, UploadFile, Path, Query
import MyAwsS3
from pydantic import BaseModel
import logging
from typing import Union

app = FastAPI()

bucket_name = 'eagleeyes'


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


@app.post("/upload-fileV2")
def upload_file(item: Item, file: UploadFile):
    hi = dummy_function(item.filename)
    MyAwsS3.upload_file(item.filename, bucket_name)
    os.remove(hi)
    return "okay"


@app.post("/upload-fileV3")
def upload_file(
        filename: str = Path
):
    MyAwsS3.upload_file(filename, bucket_name, "test.jpg")
    return "okay"


def dummy_function(item: Item):
    return "hi"