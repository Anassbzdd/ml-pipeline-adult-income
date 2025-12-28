from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import fetch_openml
import pandas as pd

data = fetch_openml(name='adult', version = 2 , as_frame=True )
x = data.data
y = data.target 

X_train , X_test , y_train , y_test = train_test_split(X,y,random_state = 0)

numeric_features = X.select_dtypes(include = ['int64','float64']).columns
numeric_transformer = StandardScaler()
categorical_features = X.select_dtypes(include = ['object']).columns
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ( 'num' , numeric_transformer , numeric_features ),
        ( 'cat' , categorical_transformer , categorical_features)
    ]
)

pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('pca', PCA()),
    ('rf',RandomForestClassifier(random_state= 0))
])

param_grid = {
    'pca__n_components' : [5,10,15],
    'rf__n_estimators' : [50,100],
    'rf__max_depth' : [None, 5, 10]
}

grid = GridSearchCV(pipe, param_grid , cv = 3)
grid.fit(X_train,y_train)

print("Best parameters:", grid.best_params_)
print("Test accuracy:", grid.score(X_test, y_test))