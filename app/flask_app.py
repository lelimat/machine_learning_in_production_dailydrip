from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from scout_apm.flask import ScoutApm # Necessary for Scout monitoring



# This is just the function we created in day 3 of this tutorial
def data_to_dummies(feature_dict, column_names):
    values = [] # list of 0/1 values to be return (new row)
    for col in column_names: # Original columns with the dummies
        for key in feature_dict.keys():
            if col.startswith(key):
                value = col[len(key)+1:]
                if feature_dict[key] == value:
                    values.append(1)
                else:
                    values.append(0)
    return values



def ctr_prediction(feature_dict, column_names, model):
    # This will create the values for the dummy
    # variables from the given columns
    values = data_to_dummies(feature_dict, column_names)
    prediction_prob = model.predict_proba(np.array([values]))    
    return prediction_prob[0][1]



def read_dataset(number_of_samples):
    import os # temp
    cwd = os.getcwd()
    print(cwd)
    if cwd.find('app') < 0:
        app_path = cwd + '/app'
    else:
        app_path = cwd

    if number_of_samples == 'all':
        data = pd.read_csv(app_path + '/data/train.gz', compression='gzip')
    else:
        data = pd.read_csv(app_path + '/data/train.gz', compression='gzip', nrows=int(number_of_samples))
    return data



def class_balance(data, downsampling):

    # Separating positive and negative clicks
    data_click_0 = data[data.click==0]
    data_click_1 = data[data.click==1]

    if data_click_0.shape[0] < data_click_1.shape[0]:
        data_minority = data_click_0
        data_majority = data_click_1
    else:
        data_minority = data_click_1
        data_majority = data_click_0
        
    minor_n = data_minority.shape[0]
    major_n = data_majority.shape[0]

    if downsampling:
        # Downsampling
        data_majority = data_majority.sample(minor_n)
    else:
        # Upsampling
        data_minority = data_minority.sample(major_n, replace=True)
  
    # Combining the new balanced classes
    data = pd.concat([data_majority, data_minority])

    # Shuffling classes to keep the samples random
    data = data.sample(data.shape[0])

    return data



#########################
#### Flask functions ####
#########################

# Creating app
app = Flask(__name__)

# Attaches ScoutApm to the Flask App
ScoutApm(app)

# Scout settings
app.config['SCOUT_MONITOR'] = True
app.config['SCOUT_KEY']     = "... your key goes here ..."
app.config['SCOUT_NAME']    = "CTR Predictor"


@app.route('/')
def home():
    return render_template('home.html')



@app.route('/train', methods=['POST'])
def train():
    number_of_samples = request.form['number_of_samples']
    balancing         = request.form['balancing']

    data = read_dataset(number_of_samples)

    if balancing == 'downsampling':
        data = class_balance(data, downsampling=True)
    else:
        data = class_balance(data, downsampling=False)

    banner_pos_dummies = pd.get_dummies(data['banner_pos'], 'banner_pos')
    banner_pos_attributes = list(banner_pos_dummies.columns)

    app_domain_dummies = pd.get_dummies(data['app_domain'], 'app_domain')
    app_domain_attributes = list(app_domain_dummies.columns)

    device_type_dummies = pd.get_dummies(data['device_type'], 'device_type')
    device_type_attributes = list(device_type_dummies.columns)

    attributes = banner_pos_attributes + app_domain_attributes + device_type_attributes

    # Independent variables (used to predict)
    X = pd.concat([banner_pos_dummies, app_domain_dummies, device_type_dummies], axis=1)

    # Dependent variables (to be predicted)
    y = data['click']

    # Setting random seed
    seed = 0
    np.random.seed(seed)

    # Training the model
    log_reg = LogisticRegression()
    log_reg.fit(X, y)

    # Saving the model
    pickle.dump(log_reg, open('CTR_pred_model.pkl', 'wb'))

    # Creating dummy variables for prediction
    dummy_cols = X.columns

    dummy_cols_dic = {}
    for col in dummy_cols:
        elems = col.split('_')
        value = elems[-1]
        col_name = '_'.join(elems[:-1])
        if col_name in dummy_cols_dic.keys():
            dummy_cols_dic[col_name].append(value)
        else:
            dummy_cols_dic[col_name] = [value]

    training_info = {'dummy_cols_dic' : dummy_cols_dic,
                     'dummy_columns'  : list(X.columns),
                     'n_samples'      : X.shape[0],
                     'n_features'     : X.shape[1]}
    
    pickle.dump(training_info, open('training_info.pkl', 'wb'))

    return predict()



@app.route('/predict')
def predict():

    # Loading the training information
    training_info = pickle.load(open('training_info.pkl', 'rb'))

    return render_template('predict.html',
                           dummy_cols_dic=training_info['dummy_cols_dic'],
                           n_samples=training_info['n_samples'],
                           n_features=training_info['n_features'])



@app.route('/get_prob/<string:banner_pos>,<string:app_domain>,<string:device_type>', methods=['GET'])
def get_probability(banner_pos, app_domain, device_type):

    feature_dict = {
        'banner_pos'  : banner_pos,
        'app_domain'  : app_domain,
        'device_type' : device_type
    }

    # Loading the training information
    training_info = pickle.load(open('training_info.pkl', 'rb'))

    # Loading the dummy columns
    dummy_cols = training_info['dummy_columns']

    # Loading the model
    log_reg = pickle.load(open('CTR_pred_model.pkl', 'rb'))

    # This will return the prediction
    pred = ctr_prediction(feature_dict, dummy_cols, log_reg)
    return jsonify({'prob': pred})



@app.route('/results', methods=['POST'])
def results():

    # Loading the training information
    training_info = pickle.load(open('training_info.pkl', 'rb'))

    # Loading the dummy columns
    dummy_cols = training_info['dummy_columns']

    # Loading the model
    log_reg = pickle.load(open('CTR_pred_model.pkl', 'rb'))

    # This will return the prediction
    pred = ctr_prediction(request.form, dummy_cols, log_reg)

    # Returning a function to render the HTML and sending
    # the variable prediction calculated by "ctr_prediction
    return render_template('results.html', prediction=pred)



if __name__ == '__main__':
  app.run(debug=True)
