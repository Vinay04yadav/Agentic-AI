from fastapi import FastAPI, Query
from client.rq_client import queue
from queues.worker import process_query



app = FastAPI()

@app.get('/')
def root():
    return {"status" : "Server is up and running"}

@app.post('/chat')
def chat(query: str = Query(..., description="The chat query of the user")):
    job = queue.enqueue(process_query, query) # returns the jobID
    return {"status" : "queued", "job_id" : job.id}
    # eg:
    # {
    # "status": "queued",
    # "job_id": "4c1304cc-cae5-4935......"
    # }

@app.get('/job-status')
def get_result(job_id: str = Query(..., description="Job ID")):
    job = queue.fetch_job(job_id=job_id)
    result = job.return_value()

    return {"result": result}

# How to run this: check main.py