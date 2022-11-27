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


@app.post("/upload-fileV3")
def upload_file(
        filename: str = Path,
        filename1: str = Path,
        file: str = Path,
        file1: str = Path
):
    MyAwsS3.upload_file(filename, bucket_name, file)
    MyAwsS3.upload_file(filename1, bucket_name, file1)
    return "okay"


def dummy_function(item: Item):
    return "hi"
