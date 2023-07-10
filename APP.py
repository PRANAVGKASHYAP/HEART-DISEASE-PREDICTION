import streamlit as st
import pandas as pd
import pickle

header = st.container()
dataset = st.container()
prediction = st.container()
features = st.container()
answers = st.container()

def find(df , model):
    ans = model.predict(y_predict)
    probabilities = model.predict_proba(y_predict)
    if(ans == 1):
        answers.write('SUCEPTABLE TO HEART DISEASE')
        answers.markdown("  :red[GO GET A HEALTH CHECKUP]")
        classes = probabilities[0]
        answers.write(classes[1])
        st.warning()

    else:
        answers.write('SAFE AND HEALTHY HEART')
        answers.markdown("  :green[GOOD JOB]")
        st.balloons()

with header:
    st.title('HEART DISEASE PREDICTION APP')

with dataset :
    st.header('Heart Disease Dataset')
    data = pd.read_csv('heart-disease.csv')
    st.write(data.head())
    st.subheader('1:HAS DISEASE ')
    st.subheader('0:NO DISEASE')
    plot_df = pd.DataFrame(data['target'].value_counts())
    st.bar_chart(plot_df)

    

with prediction:
    st.header('Predictions from the model')
    st.subheader('use the slidert to give the model inputs')

    sel_col , disp_col = st.columns(2)
    age = sel_col.slider('WHAT IS YOUR AGE',min_value=0,max_value=100,value=20,step=1)
    sex = sel_col.selectbox('1:Male , 0:Female',(1,0))
    cp = sel_col.selectbox('SELECT THE TYPE OF CHEST PAIN',(0,1,2,3))
    trestbps = sel_col.slider('trestbps',min_value=0,max_value=200,value=100,step=10)
    chol = sel_col.slider('Cholestrol',min_value=0,max_value=300,value=100,step=10)
    fbs = sel_col.selectbox('fbs 1:Yes , 0:No',(1,0))
    restecg = sel_col.selectbox('restecg 1/0',(1,0))
    thalach = sel_col.slider('thalach',min_value=0,max_value=250,value=100,step=10)
    exang = sel_col.selectbox('exang 1/0',(1,0))
    oldpeak = sel_col.text_input('enter approximate value')
    slope = sel_col.text_input('slope : 0/1/2')
    ca = sel_col.text_input('ca : 0/1/2')
    thal = sel_col.selectbox('thal:',(1,2,3))

index = range(100)

st.header('THE USER INPUT FOR PREDICTION')
y_predict = pd.DataFrame({
    'age': age,
    'sex': sex,
    'cp': cp,
    'trestbps': trestbps,
    'chol': chol,
    'fbs': fbs,
    'restecg': restecg,
    'thalach': thalach,
    'exang': exang,
    'oldpeak': oldpeak,
    'slope': slope,
    'ca': ca,
    'thal': thal
},index=[0])

st.write(y_predict)

# making predictions 
loaded_model = pickle.load(open('heart_disease_clf','rb'))
ans = loaded_model.predict(y_predict)
probabilities = loaded_model.predict_proba(y_predict)



# creating a button to find the result
if st.button('PREDICT'):
    find(y_predict , loaded_model)  
    st.header('THE PREDICTED RESULT SUMMARY')  
    st.write(ans)
    st.write(probabilities)


# creating a feature dictionary to plot feature importance graph
with features:
    st.header('Feature Importance Graph')
    feature_dict = dict(zip(data.columns,list(loaded_model.coef_[0])))
    feature_df = pd.DataFrame(feature_dict , index=[0])
    st.bar_chart(feature_df.T)

if st.button('CLEAR'):
    features.empty()
    