from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root_controller():
    """
    endpoint to check if the server is running
    """

    return {"message": "Hello World"}

@app.get("/health")
def health_controller():
    """
    endpoint to check if the server is running
    """
    return {"message": "OK"}