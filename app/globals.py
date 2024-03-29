import os
import redis
import json
import time


# Initialize Redis
redis_url = os.getenv('REDIS_URL')
redis_client = redis.from_url(redis_url, decode_responses=True, ssl_cert_reqs=None)


def store_user_session_data(session_id, data):
    """
    Store user session data in Redis.
    """

    data_json = json.dumps(data)
    redis_client.set(f"session:{session_id}", data_json)

    # Update the last access time
    redis_client.set(f"session:timestamp:{session_id}", int(time.time()))


def get_user_session_data(session_id):
    """
    Retrieve user session data from Redis.
    """

    data_json = redis_client.get(f"session:{session_id}")

    # Update the last access time before returning the data
    redis_client.set(f"session:timestamp:{session_id}", int(time.time()))

    return json.loads(data_json)
