from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Suppose you have all your static files (including index.html) in a folder named "static"
app.mount("/", StaticFiles(directory="static", html=True), name="static")
