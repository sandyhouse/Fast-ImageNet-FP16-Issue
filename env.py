import os


def dist_env():
    """
    Return a dict of all variable that distributed training may use.
    NOTE: you may rewrite this function to suit your cluster environments.
    """
    #trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
    #num_trainers = int(os.getenv("PADDLE_TRAINERS_NUM", "1"))

    ## - PADDLE_TRAINER_ENDPOINTS means nccl2 mode.
    #trainer_endpoints = os.getenv("PADDLE_TRAINER_ENDPOINTS")
    #trainer_endpoints = trainer_endpoints.split(',')

    # For launch.py
    trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
    num_trainers = int(os.getenv("PADDLE_TRAINERS_NUM", "0"))
    current_endpoint = os.getenv("PADDLE_CURRENT_ENDPOINT", "127.0.0.1:6170")
    trainer_endpoints = os.getenv("PADDLE_TRAINER_ENDPOINTS", "127.0.0.1:6170")
    trainer_endpoints = trainer_endpoints.split(',')

    #current_endpoint = os.getenv("PADDLE_CURRENT_ENDPOINT")

    assert num_trainers == len(trainer_endpoints)
    
    return {
        "trainer_id": trainer_id,
        "num_trainers": num_trainers,
        "current_endpoint": current_endpoint,
        "trainer_endpoints": trainer_endpoints
    }
