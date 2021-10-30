from rest_framework.views import APIView
from rest_framework.response import Response
import pickle
from pathlib import Path
import os
class Test(APIView):
    def post(self, request, format=None):
        BASE_DIR = Path(__file__).resolve().parent
        path = os.path.join(BASE_DIR, "sars_cov2_pred_gaussien_nb.pkl")

        with open(path, 'rb') as f:
            loaded_model = pickle.load(f)
            data = request.data 

            x1 = [data['pulse'],data['sys'],data['temperature'],data['smoker'],data['rr'],data['wheezes'],data['dia'],data['labored_respiration'], data['high_risk_exposure_occupation'],data['days_since_symptom_onset'],data['loss_of_smell'], data['muscle_sore'],data['fatigue'],data['diarrhea']]
            x = [x1]
            result = loaded_model.predict(x)
            
            return Response({
                'result': result
            })