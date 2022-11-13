from fastapi import FastAPI, File, UploadFile
from typing import List
import os
import MyAwsS3

app = FastAPI()

bucket_name = ""

@app.get("/")
def read_root():
  return { "Hello": "World" }

@app.get("/fastapi-test")
def test_api():
    return "hi"

@app.post("/upload-file")
def upload_file(file_name:str):
    MyAwsS3.upload_file(file_name, bucket_name)