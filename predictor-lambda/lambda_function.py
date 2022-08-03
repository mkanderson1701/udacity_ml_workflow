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