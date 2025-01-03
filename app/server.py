from fastapi import FastAPI, File, UploadFile
import uvicorn
import os
from fastapi import FastAPI, HTTPException

from retriever import process_file_fastapi

# 작업 디렉토리 변경
app_dir = os.path.dirname(__file__)  # 현재 파일(server.py)의 디렉토리
os.chdir(app_dir)
print(f"Working directory changed to: {os.getcwd()}")

# FastAPI 앱 생성
app = FastAPI()


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    # FastAPI 파일 처리 호출
    await process_file_fastapi(file)
    print(f"Success save file_name is: {file.filename}")
    return {"file_path": file.filename, "chain_status": "created"}


# FastAPI 서버 실행 (Streamlit 실행 코드 아래에 추가)
if __name__ == "__main__":
    print(f"Current working directory: {os.getcwd()}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
