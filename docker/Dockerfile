FROM python:3.7.3

WORKDIR /workspace

# install python library
RUN pip install numpy pandas scikit-learn
RUN pip install mxnet autogluon bokeh matplotlib
RUN pip install streamlit
RUN pip install kaggle

# add entrypoint script
ADD ./entry.sh /workspace
