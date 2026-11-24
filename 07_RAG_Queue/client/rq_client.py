from redis import Redis
from rq import Queue

queue = Queue(connection=Redis(
    host="localhost",
    port="6379"
))

# A queue lets one part of a program submit work without doing that work immediately. The job waits in the queue until a separate worker picks it up. This is useful for slow tasks—like processing a document or generating embeddings—because the part responding to the user doesn’t have to wait for the task to finish. Queues also separate the job producer from the worker and can buffer work when many jobs arrive at once.

# In this setup, Redis is the shared place where queued jobs are stored, and RQ is the Python library that lets your code put jobs there and lets worker processes retrieve and run them. The work happens in a background process, so it’s asynchronous from the client’s perspective—but this is different from Python’s async/await.

# We will submit a function with queue.enqueue(...) and need a separate RQ worker process running to execute it.