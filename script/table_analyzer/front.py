import streamlit as st
import pandas as pd
from analyzer import TableAnalyzerFactory
import os
from datetime import datetime as dt

# autogluon(TODO : 将来的には別ファイルに切り出す。UIページをautogluonと依存させたくないため)
import autogluon as ag

# scikit-learn
from sklearn.model_selection import train_test_split 

def main():
    # set default config
    # TODO : enable to read config from json files
    config = {
        "output_base_dir": os.path.join("script", "output")
    }
    params = {
        "hp_tune"   : True,
        "xgboost"   : {},
        "autogluon" : {
            "hyperparameters": {
                "time_limits"    : 2*60,
                "num_trials"     : 5,
                "search_strategy": "skopt",
                "NN": { # specifies non-default hyperparameter values for neural network models
                    'num_epochs'   : 10, # number of training epochs (controls training time of NN models)
                    'learning_rate': ag.space.Real(1e-4, 1e-2, default=5e-4, log=True), # learning rate used in training (real-valued hyperparameter searched on log-scale)
                    'activation'   : ag.space.Categorical('relu', 'softrelu', 'tanh'), # activation function used in NN (categorical hyperparameter, default = first entry)
                    'layers'       : ag.space.Categorical([100],[1000],[200,100],[300,200,100]),
                    'dropout_prob' : ag.space.Real(0.0, 0.5, default=0.1), # dropout probability (real-valued hyperparameter)
                },
                "GBM": { # specifies non-default hyperparameter values for lightGBM gradient boosted trees
                    'num_boost_round': 100, # number of boosting rounds (controls training time of GBM models)
                    'num_leaves'     : ag.space.Int(lower=26, upper=66, default=36), # number of leaves in trees (integer hyperparameter)
                }
            }
        } 
    }
    
    # set common sidebar
    page       = st.sidebar.radio("move page", ["train", "inference", "kaggle"])
    model_type = st.sidebar.selectbox("select model type", ["autogluon", "xgboost"])
    if page == "train":
        render_train_page(model_type, config, params)
    elif page == "inference":
        render_infer_page(model_type, config)
    elif page == "kaggle":
        render_kaggle_page(model_type, config)
    else:
        st.write("error page")

def render_train_page(model_type, config, params):
    st.title("Train Page")
    
    uploaded_file = st.file_uploader("Choose a csv file", type="csv")
    if uploaded_file is not None:
        # read csv file and display columns
        df = pd.read_csv(uploaded_file)
        
        # display sample data
        st.write("display sample data")
        st.write(df.head(10))
        
        # select columns
        label = st.radio("select label column", df.columns, index=0)
        params["label"] = label
        if st.button("Start Train"):
            analyzer = TableAnalyzerFactory.get_instance(model_type=model_type, config=config)
            start_train(analyzer, df, params)
            st.success("Model Training is done!")
        else:
            print("please push button")

def start_train(analyzer, df, params):
    train_data, val_data = train_test_split(df, test_size=0.2)
    analyzer.train(train_data=train_data, val_data=val_data, params=params)

def render_infer_page(model_type, config):
    st.title("Inference Page")
    
    uploaded_file = st.file_uploader("Choose a csv file", type="csv")
    if uploaded_file is not None:
        # read csv file and display columns
        df = pd.read_csv(uploaded_file)
        
        # display sample data
        st.write("display sample data")
        st.write(df.head(10))
        
        # get model list from predict result directory
        analyzer             = TableAnalyzerFactory.get_instance(model_type=model_type, config=config)
        train_folder         = analyzer.get_output_folders()[0]
        model_folder_name    = st.radio("select model", os.listdir(train_folder), index=0)
        config["model_path"] = os.path.join(train_folder, model_folder_name)
        if st.button("Start Predict"):
            analyzer.load(path=config["model_path"])
            result = analyzer.predict(data=df)
            
            # display result
            st.write("display result")
            label = analyzer.get_label_name()
            df[label] = result
            st.write(df)
            
            # save result
            predict_folder = analyzer.get_output_folders()[1]
            os.makedirs(predict_folder, exist_ok=True)
            csv_path = os.path.join(predict_folder, dt.now().strftime('%Y%m%d%H%M%S') + "_result.csv")
            df.to_csv(csv_path)
            st.success("Inference is done!")

def render_kaggle_page(model_type, config):
    st.title("Kaggle Page")
    
    from kaggle_wrapper import KaggleWrapper
    
    uploaded_file = st.file_uploader("Choose a result csv file", type="csv")
    if uploaded_file is not None:
        # read csv file and display columns
        df = pd.read_csv(uploaded_file)
        
        # display sample data
        st.write("display sample data")
        st.write(df.head(10))
        
        # select exported columns
        export_columns = st.multiselect("select the columns of export", df.columns)
        competition_id = st.text_input("Input competition_id of Kaggle")
        description    = st.text_input("Input competition description")
        if len(competition_id) > 0 and len(description) > 0 and st.button("Export result to Kaggle page"):
            # save csv file
            df = df.loc[:, export_columns].set_index(export_columns[0])
            #csv_dir  = os.path.dirname(uploaded_file)
            csv_path = os.path.join("submit.csv")
            df.to_csv(csv_path)
            
            # submit csv
            wrapper = KaggleWrapper()
            wrapper.submit_result(csv_path=csv_path, competition_id=competition_id, description=description)
            st.success("Success to submit result!")
            
            # delete submit file(TODO : 将来的にfile_uploaderからpathが変えるようになったら、submitファイルは残すようにする)
            os.remove(csv_path)

if __name__ == "__main__":
    main()
