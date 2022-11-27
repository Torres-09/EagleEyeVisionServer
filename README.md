## 실행방법
- 터미널에 아래 메세지를 입력한다.
- uvicorn main:app --reload
- 기본 포트 8000번에서 실행된다.
- python3 -m uvicorn main:app --reload (위의 시도가 안된다면 이것으로 시도 해 볼 것)

## configure 설정
- install awscli
- aws configure 명령어로
- aws 정보들 입력해준다.

## 기본 세팅
- pip install fastapi
- pip install "uvicorn[standard]"
- pip3 install boto3
- apt-get install python3
- pip install python-multipart