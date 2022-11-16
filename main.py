from fastapi import FastAPI, File, UploadFile
import MyAwsS3
from pydantic import BaseModel

app = FastAPI()

bucket_name = 'makeanything'

class Item(BaseModel):
    filename : str

@app.get("/")
def read_root():
  return { "Hello": "World" }

@app.get("/fastapi-test")
def test_api():
    return "hi"

@app.post("/fastapi-test2")
def test_api(example:str):
    return example

@app.post("/upload-file")
def upload_file(item : Item):
    MyAwsS3.upload_file(item.filename, bucket_name)