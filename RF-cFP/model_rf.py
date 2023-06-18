from sklearn.ensemble import RandomForestClassifier
from deepchem.models.sklearn_models import SklearnModel

model_dir = './models'
def rf_model_builder(model_params, model_dir=model_dir):
      sklearn_model = RandomForestClassifier(**model_params)
      return SklearnModel(sklearn_model, model_dir)