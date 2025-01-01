import os
import shutil
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from retriever import handling_file

WATCH_FOLDER = "./uploads"
COMPLETED_FOLDER = "./completed"
os.makedirs(WATCH_FOLDER, exist_ok=True)
os.makedirs(COMPLETED_FOLDER, exist_ok=True)


class FileEventHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        new_file_path = event.src_path
        print(f"새 파일 감지됨: {new_file_path}")
        process_file(new_file_path)


def process_file(file_path):
    try:
        # 파일을 열어 파일 객체로 처리
        with open(file_path, "rb") as file:
            handling_file(file)

        # 처리 완료된 파일을 완료 폴더로 이동
        completed_path = os.path.join(COMPLETED_FOLDER, os.path.basename(file_path))
        shutil.move(file_path, completed_path)
        print(f"파일 처리 완료 및 이동: {completed_path}")

    except Exception as e:
        print(f"파일 처리 중 오류 발생: {e}")


def start_file_watch():
    observer = Observer()
    event_handler = FileEventHandler()
    observer.schedule(event_handler, WATCH_FOLDER, recursive=False)
    observer.start()
    print("폴더 감시가 시작되었습니다.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
