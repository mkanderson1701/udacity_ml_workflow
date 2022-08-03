# this is not a module or valid script, just a dump from lambda.
# there are three functions with the same name.


"""
Source for lambda serializer

serializeImageData

"""

import json
import boto3
import base64

s3 = boto3.client('s3')

def lambda_handler(event, context):
    """A function to serialize target data from S3"""

    # Get the s3 address from the Step Function event input
    key = event['s3_key']
    bucket = event['s3_bucket']

    # Download the data from s3 to /tmp/image.png
    s3 = boto3.client('s3')
    s3.download_file(bucket, key, '/tmp/image.png')

    # We read the data from a file
    with open("/tmp/image.png", "rb") as f:
        image_data = base64.b64encode(f.read())

    # Pass the data back to the Step Function
    print("Event:", event.keys())
    return {
        'statusCode': 200,
        'body': {
            "image_data": image_data,
            "s3_bucket": bucket,
            "s3_key": key,
            "inferences": []
        }
    }


"""
Source for lambda predictor

runPrediction

"""

import json
import sagemaker
import base64
from sagemaker.serializers import IdentitySerializer

# Fill this in with the name of your deployed model
ENDPOINT = 'image-classification-2022-08-03-01-56-12-665'

def lambda_handler(event, context):

    # event is wrapped in body dict
    body_event = event['body']

    image = base64.b64decode(body_event['image_data'])

    # Instantiate a Predictor
    predictor = sagemaker.predictor.Predictor('image-classification-2022-08-03-12-39-49-455')

    # For this model the IdentitySerializer needs to be "image/png"
    predictor.serializer = IdentitySerializer("image/png")

    # Make a prediction:
    inferences = predictor.predict(image)

    # We return the data back to the Step Function    
    body_event["inferences"] = inferences.decode('utf-8')
    
    return {
        'statusCode': 200,
        'body': json.dumps(body_event)
    }


"""
Source for lambda confidence check

filterLowConfidence

"""

import json


THRESHOLD = .93

class LowConfidence(Exception):
    def __init__(self, message):
        super().__init__(message)


def lambda_handler(event, context):

    # event is wrapped in body dict
    body_event = json.loads(event['body'])

    # I tried using the bytes deserializer on the predictor but I still get a
    # bytes / string representation of a list... dealing with it
    inferences = body_event['inferences'].strip('[]').split(', ')

    # Check if any values in our inferences are above THRESHOLD
    meets_threshold = any(float(x) > THRESHOLD for x in inferences)

    # If our threshold is met, pass our data back out of the
    # Step Function, else, end the Step Function with an error
    if meets_threshold:
        pass
    else:
        raise LowConfidence("THRESHOLD_CONFIDENCE_NOT_MET")
        # I realize this block will never execute, but I might handle it differently.
        return {
            'statusCode': 418,
            'body': body_event
        }
        
    return {
        'statusCode': 200,
        'body': body_event
    }