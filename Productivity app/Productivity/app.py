from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

# setup application
app = Flask(__name__,template_folder='template')

def prediction(lst):
    filename = 'model/predictor.pickle'
    with open(filename, 'rb') as file:
        x = pd.array([])
        dic= {'targeted_productivity': [0.6], 'smv':[22.53], 'wip':[708.0], 'over_time':[5040], 'incentive':[12],'idle_time': [3.4], 'idle_men': [0], 'no_of_workers':[42],'department':'sweing','day':'Tuesday','quarter':'Quarter4'}
        x = pd.DataFrame(dic)
        model = pickle.load(file)
   #pred_value = model.predict([lst])
    pred_value = model.predict(x)
    return pred_value

@app.route('/', methods=['POST', 'GET'])
def index():
    
    pred_value = 0
    if request.method == 'POST':
        # Create a dictionary to hold input values
        input_data = {
            'targeted_productivity': float(request.form['targeted_productivity']),
            'svm': float(request.form['svm']),
            'wip': float(request.form['wip']),
            'over_time': float(request.form['over_time']),
            'incentive': float(request.form['incentive']),
            'idle_time': float(request.form['idle_time']),
            'idle_men': int(request.form['idle_men']),
            'no_of_worker': int(request.form['no_of_workers']),
            'department': request.form['department'],
            'day': request.form['day'],
            'quarter': request.form['quarter']
        }

        # Prepare the feature list
        feature_list = [
            input_data['targeted_productivity'],
            input_data['svm'],
            input_data['wip'],
            input_data['over_time'],
            input_data['incentive'],
            input_data['idle_time'],
            input_data['idle_men'],
            input_data['no_of_worker'],
            input_data['department'],
            input_data['day'],
            input_data['quarter']
        ]
        keys = ['targeted_productivity','svm','wip','over_time','incentive','idle_time','idle_men','no_of_worker','department','day','quarter']
        vdata_dict = dict(zip(keys, feature_list))
        
      
        

        print(vdata_dict)
        pred_value = prediction(vdata_dict)
        pred_value = np.round(pred_value[0], 3)

    return render_template('index.html', pred_value=pred_value)


if __name__ == '__main__':
    app.run(debug=True)