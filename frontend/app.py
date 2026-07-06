import streamlit as st
import requests

BASE_URL = 'http://127.0.0.1:8000'

st.title('Motor Transport Detection')

model_version = st.selectbox(
    'Choose model for prediction',
    [
        'Standard',
        'More accurate, but less fast'
    ]
)

uploaded_files = st.file_uploader(
    'Choose a directory that contains all the images you want to make prediction of.',
    type=['image/*'],
    accept_multiple_files='directory'
)

if st.button('Make prediction') and uploaded_files:
    files = []

    for file in uploaded_files:
        files.append(
            [
                'files',
                [
                    file.name,
                    file,
                    file.type
                ]
            ]
        )

    response = requests.post(
        f'{BASE_URL}/prediction/',
        data={'model_version': model_version},
        files=files
    )

    st.write(response.json())