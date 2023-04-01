import os
import torch
import logging
import sentry_sdk
from sentry_sdk.integrations.logging import LoggingIntegration
from ultralyticsplus import YOLO

import traceback
from dotenv import load_dotenv
from PIL import Image
import numpy as np
from time import time

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"


def init():
    global model
    try:
        start_time = time()
        load_dotenv()

        ##############################
        # Initialize Sentry
        ##############################
        sentry_dsn = os.getenv('SENTRY_DSN')
        sentry_logging = LoggingIntegration(
            level=logging.INFO,        # Capture info and above as breadcrumbs
            event_level=logging.INFO  # Send errors as events
        )
        sentry_sdk.init(
            dsn=sentry_dsn,
            max_breadcrumbs=50,
            integrations=[
                sentry_logging,
            ],
            traces_sample_rate=1.0,
        )
        logging.info('Sentry initialized')

        ##############################
        # Load the model / pipeline
        ##############################
        model = YOLO('keremberke/yolov8m-table-extraction')
        # set model parameters
        model.overrides['conf'] = 0.45  # NMS confidence threshold
        model.overrides['iou'] = 0.45  # NMS IoU threshold
        model.overrides['agnostic_nms'] = False  # NMS class-agnostic
        model.overrides['max_det'] = 1000  # maximum number of detections per image
        logging.info('Model loaded')

        ##############################
        # Move model to GPU
        ##############################
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        logging.info('Initialization complete in {} seconds'.format(time() - start_time))

    except Exception as e:
        logging.error(traceback.format_exc())
        logging.error(e)
        raise e


def inference(model_inputs: dict) -> dict:
    start_time = time()
    try:
        global model

        ######################################
        # Parse arguments
        ######################################
        ping = model_inputs.get('ping', None)
        if ping is not None:
            # logging.info('Ping received')
            return {
                'message': 'pong',
                'total_time': f'{time() - start_time:.2f}',
            }

        inputs_list = model_inputs.get('inputs', None)
        if inputs_list is None:
            return {
                'message': "No inputs provided"
            }

        ######################################
        # Run the model (return list of 1, and 0s)
        ######################################
        t0 = time()
        inputs = [np.array(x, dtype=np.uint8) for x in inputs_list]
        table_dets = model.predict(inputs)
        logging.info(f"Model run in {time() - t0:.2f} seconds")
        preds = [1 if len(dets.boxes) > 0 else 0 for dets in table_dets]

        ######################################
        # Return the results as a dictionary
        ######################################
        return {
            'message': 'Success',
            'result': preds,
            'time': f'{time() - t0:.2f}',
            'total_time': f'{time() - start_time:.2f}',
            'device': 'YOLO'
        }

    except Exception as e:
        logging.error(traceback.format_exc())
        logging.error(e)
        return {
            'error': str(traceback.format_exc()) + str(e)
        }
