from flask import Flask, render_template, request
from sklearn.model_selection import KFold

app = Flask(__name__)


@app.route('/')
def index():
    return render_template("crop.html")


@app.route('/', methods=['post'])
def getvalue():
    state = request.form['state_name']
    district = request.form['district_name']
    district = district.upper()
    crop = request.form['crop']
    season = request.form['season']
    area = request.form['area']
    area_float = float(area)
    year = request.form['year']
    year_int = int(year)
    import pandas as pd
    import numpy as np
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import RobustScaler
    from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
    from sklearn.ensemble import RandomForestRegressor
    import os
    os.chdir(r"C:\Users\Hp\Downloads\indian-farming-prediction-master")
    crop_data = pd.read_csv("crop_modified.csv")
    crop_data = crop_data.dropna()
    crop_data['State_Name'] = crop_data['State_Name'].str.rstrip()
    crop_data['Season'] = crop_data['Season'].str.rstrip()
    a = crop_data[crop_data['State_Name'] == state]
    b = a[a['District_Name'] == district]
    c = b[b['Season'] == season]
    f = c[c['Crop'] == crop]['Crop_Year']
    x = c[c['Crop'] == crop]['Area']
    y = c[c['Crop'] == crop]['Production']
    from pandas import DataFrame
    variables = {'Crop_Year': f, 'Area': x, 'Production': y}
    final = DataFrame(variables, columns=['Crop_Year', 'Area', 'Production'])
    X = final[['Crop_Year', 'Area']]
    Y = final['Production']

    class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
        def __init__(self, base_models, meta_model, n_folds=5):
            self.base_models = base_models
            self.meta_model = meta_model
            self.n_folds = n_folds

        # We again fit the data on clones of the original models
        def fit(self, X, y):
            self.base_models_ = [list() for x in self.base_models]
            self.meta_model_ = clone(self.meta_model)
            kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

            # Train cloned base models then create out-of-fold predictions
            # that are needed to train the cloned meta-model
            out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
            for i, model in enumerate(self.base_models):
                for train_index, holdout_index in kfold.split(X, y):
                    print(X.columns)
                    instance = clone(model)
                    self.base_models_[i].append(instance)
                    instance.fit(X[train_index], y[train_index])
                    y_pred = instance.predict(X[holdout_index])
                    out_of_fold_predictions[holdout_index, i] = y_pred

            # Now train the cloned  meta-model using the out-of-fold predictions as new feature
            self.meta_model_.fit(out_of_fold_predictions, y)
            return self

        # Do the predictions of all base models on the test data and use the averaged predictions as
        # meta-features for the final prediction which is done by the meta-model
        def predict(self, X):
            meta_features = np.column_stack([
                np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
                for base_models in self.base_models_])
            return self.meta_model_.predict(meta_features)

    class StackedAveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
        def __init__(self, models):
            self.models = models

        # we define clones of the original models to fit the data in
        def fit(self, X, y):
            self.models_ = [clone(x) for x in self.models]

            # Train cloned base models
            for model in self.models_:
                model.fit(X, y)

            return self

        # Now we do the predictions for cloned models and average them
        def predict(self, X):
            predictions = np.column_stack([
                model.predict(X) for model in self.models_
            ])
            return np.mean(predictions, axis=1)

    lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1))
    ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
    KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
    # from mlxtend.regressor import StackingRegressor
    # stack = StackingRegressor(regressors=[ENet, KRR],  meta_regressor=lasso)
    # model1=stack.fit(X,Y)
    # prod2 = model.predict([[year_int, area_float]])
    averaged_models = StackedAveragingModels(models=(KRR, lasso))

    # import pickle
    # pickle.dump(averaged_models,open('model.pkl','wb'))
    # model = pickle.load(open('model.pkl','rb'))
    model = averaged_models.fit(X, Y)
    prod2 = model.predict([[year_int, area_float]])
    prod2 = abs(prod2)
    print("Prediction is: ", prod2)
    yld = prod2 / area_float
    return render_template("crop.html", pr=prod2, yl=yld)


if __name__ == "__main__":
    app.run()
