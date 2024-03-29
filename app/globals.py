import os
import redis
import json

# Initialize Redis
redis_url = os.getenv('REDIS_URL')
redis_client = redis.from_url(redis_url, decode_responses=True, ssl_cert_reqs=None)


def store_user_session_data(session_id, data):
    """
    Store user session data in Redis.
    """

    data_json = json.dumps(data)
    redis_client.set(f"session:{session_id}", data_json)


def get_user_session_data(session_id):
    """
    Retrieve user session data from Redis.
    """

    data_json = redis_client.get(f"session:{session_id}")
    return json.loads(data_json)


def delete_user_session_data(session_id):
    """
    Delete user session data from Redis.
    """
    redis_client.delete(f"session:{session_id}")
