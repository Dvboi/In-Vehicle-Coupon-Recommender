from flask import Flask,render_template,url_for,request
from sklearn import metrics
from catboost import CatBoostClassifier
import csv
import io
import pandas as pd 

model = CatBoostClassifier().load_model("Final_cb")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/prediction',methods=['GET','POST'])
def prediction():

    ''' Receive a single data-point and make prediction ''' 

    if request.method=='POST':
        d = request.form.to_dict()
        input_array = []
        # we need to take features of each data-point in the order we trained and then pass it to the model
        for col in model.feature_names_:
            value = d.get(col)
            input_array.append(value)

        # send the data point to model
        y_pred = model.predict(input_array)
    return render_template('prediction.html',ans=y_pred)

@app.route('/evaluation',methods=['GET','POST'])
def evaluation():

    ''' receive flask-Filestorage from webpage and convert to CSV files and
    return evaluations
    '''

    # Reference to read filestorage as csv data :- https://stackoverflow.com/a/56393796
    if request.method=='POST':
        test_data = request.files['test_file'].stream.read()
        y_test = request.files['test_label'].stream.read()
        if not test_data and y_test:
            return "Both files not submitted"

        stream1 = io.StringIO(test_data.decode("UTF8"), newline=None)
        stream2 = io.StringIO(y_test.decode("UTF8"), newline=None) 

        test_data = pd.read_csv(stream1)
        y_test = pd.read_csv(stream2)


    # prediction
    y_pred = model.predict(test_data)
    y_pred_proba = model.predict_proba(test_data)[:,1]

    # evaluation
    f1 = metrics.f1_score(y_test.astype(int),y_pred)
    auc = metrics.roc_auc_score(y_test.astype(int),y_pred_proba)

    return render_template('evaluation.html',f1=f1,auc=auc)

if __name__ == "__main__":
    app.run(debug=True)
