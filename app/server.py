from fastapi import FastAPI, File, UploadFile
import uvicorn
import os

# FastAPI 앱 생성
app = FastAPI()

# 업로드된 파일을 저장할 경로
UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_location, "wb") as f:
        f.write(await file.read())
    return {"file_path": file_location}


# FastAPI 서버 실행 (Streamlit 실행 코드 아래에 추가)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
