# views.py
import pandas as pd
from django.http import JsonResponse
from rest_framework.decorators import api_view
from pycaret.classification import predict_model, load_model
from .serializers import PredictSerializer
import os
import pickle

# Path to the saved model and encoders
model_path = os.path.join('mapp', 'models', 'LinearDiscriminantAnalysis')
preprocessor_path = os.path.join('mapp', 'models', 'preprocessor.pkl')

# Load the model
model = load_model(model_path)

# Load the preprocessor (encoder)
with open(preprocessor_path, 'rb') as f:
    preprocessor = pickle.load(f)

@api_view(['POST'])
def predict(request):
    serializer = PredictSerializer(data=request.data)
    if serializer.is_valid():
        features = serializer.validated_data

        # Convert features to DataFrame
        features_df = pd.DataFrame([features])

        # Transform the data using the preprocessor
        transformed_features = preprocessor.transform(features_df)

        # Convert transformed features back to DataFrame with appropriate column names
        transformed_df = pd.DataFrame(transformed_features, columns=features_df.columns)

        # Make prediction
        prediction = predict_model(model, data=transformed_df)
        
        # Extract prediction and prediction_score
        predicted_label = prediction['prediction_label'][0]
        prediction_score = float(prediction['prediction_score'][0])  # Convert to float
        
        # Determine 'keterangan' based on prediction_label and prediction_score
        keterangan = calculate_keterangan(predicted_label, prediction_score)
        
        return JsonResponse({
            'prediction': predicted_label,
            'prediction_score': prediction_score,
            'keterangan': keterangan
        })
    
    return JsonResponse({'error': 'Invalid input', 'details': serializer.errors}, status=400)

def calculate_keterangan(predicted_label, prediction_score):
    keterangan = 'Tidak Diketahui'
    if predicted_label == 'layak':
        if 0 <= prediction_score <= 0.143:
            keterangan = 'Resiko Tinggi'
        elif 0.1431 <= prediction_score <= 0.2856:
            keterangan = 'Sangat Lemah'
        elif 0.2857 <= prediction_score <= 0.4284:
            keterangan = 'Lemah'
        elif 0.4285 <= prediction_score <= 0.5712:
            keterangan = 'Meragukan'
        elif 0.5713 <= prediction_score <= 0.714:
            keterangan = 'Cukup'
        elif 0.7141 <= prediction_score <= 0.8568:
            keterangan = 'Kuat'
        elif 0.8569 <= prediction_score <= 1:
            keterangan = 'Sangat Kuat'
    elif predicted_label == 'tidak layak':
        if 0 <= prediction_score <= 0.143:
            keterangan = 'Sangat Kuat'
        elif 0.1431 <= prediction_score <= 0.2856:
            keterangan = 'Kuat'
        elif 0.2857 <= prediction_score <= 0.4284:
            keterangan = 'Cukup'
        elif 0.4285 <= prediction_score <= 0.5712:
            keterangan = 'Meragukan'
        elif 0.5713 <= prediction_score <= 0.714:
            keterangan = 'Lemah'
        elif 0.7141 <= prediction_score <= 0.8568:
            keterangan = 'Sangat Lemah'
        elif 0.8569 <= prediction_score <= 1:
            keterangan = 'Resiko Tinggi'
    return keterangan
