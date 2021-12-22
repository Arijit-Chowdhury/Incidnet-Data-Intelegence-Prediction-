
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from flask import Flask, request, jsonify
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import json
import nltk
import re

filename = 'finalized_model.sav'
def clean_text(text):
    # remove backslash-apostrophe 
    text = re.sub("\'", "", text) 
    # remove everything except alphabets 
    text = re.sub("[^a-zA-Z]"," ",text) 
    # remove whitespaces 
    text = ' '.join(text.split()) 
    # convert text to lowercase 
    text = text.lower() 
    return text
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    no_stopword_text = [w for w in text.split() if not w in stop_words]
    return ' '.join(no_stopword_text)






app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    print (request.is_json)
    content = request.get_json()
    q=(content['Description'])
    print (q)
    loaded_model = pickle.load(open(filename, 'rb'))
    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=1000)
    q = clean_text(q)
    q = remove_stopwords(q)
    loaded_vectorizer = pickle.load(open('vectorizer.pickle', 'rb'))
    q_vec = loaded_vectorizer.transform([q])
    prediction = loaded_model.predict(q_vec)
    loaded_service = pickle.load(open('finalized_model_service.sav', 'rb'))
    service=loaded_service.predict(q_vec)
    output = '[{"Assignment Group":"'+prediction[0]+'","Service":"'+service[0]+'"}]'
    output=json.loads(output)
    import pypyodbc    
    from datetime import datetime
    connection = pypyodbc.connect('Driver={SQL Server Native Client 11.0};Server=.;Database=SystemData;Trusted_Connection=yes;')    
    alert_desc = content['Description']   
    login_time= datetime.now()
    Source = request.remote_addr
    assignment_group = prediction[0]
    service_name = service[0]
    cursor = connection.cursor()   
    SQLCommand = ("INSERT INTO model_access_log(access_time,alert_description,source_ip,assingment_group,service) VALUES(?,?,?,?,?)")    
    Values = [login_time,alert_desc,Source,assignment_group,service_name]   
    
    #Processing Query    
    cursor.execute(SQLCommand,Values)     
    #Commiting any pending transaction to the database.    
    connection.commit()    
    #closing connection    
    print("Data Successfully Inserted")   
    connection.close()    

    return jsonify(output)
    

if __name__ == "__main__":
    #app.run(debug=True)
    #from waitress import serve
    #serve(app, host="0.0.0.0", port=5080)
    app.run(host='0.0.0.0', port= 5000)
