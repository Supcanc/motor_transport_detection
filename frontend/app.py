import streamlit as st
import requests
from pathlib import Path
import os
import shutil

BASE_URL = 'http://127.0.0.1:8000'

st.title('Motor Transport Detection')

model_version = st.selectbox(
    'Choose model for prediction',
    [
        'Standard',
        'More accurate, but less fast'
    ]
)

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

uploaded_files = st.file_uploader(
    'Choose a directory that contains all the images you want to make prediction of',
    type=['image/*'],
    accept_multiple_files='directory',
    key=f"uploader_{st.session_state.uploader_key}"
)

if st.button("Clear uploader"):
    st.session_state.uploader_key += 1
    st.rerun()

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

    prediction_response = requests.post(
        f'{BASE_URL}/prediction/',
        data={'model_version': model_version},
        files=files
    )

    if prediction_response.status_code == 200:
        preds = prediction_response.json()['results']

        annots_dir_path = Path('annots')
        annots_dir_path.mkdir(parents=True, exist_ok=True)

        for file_path in preds.keys():
            file_name = Path(file_path).name
            file_stem = Path(file_path).stem
            with open(os.path.join(annots_dir_path, file_stem + '.txt'), 'w') as f:
                f.write('x y x y confidence class_id\n')
                for obj in preds[file_path]:
                    f.write(f'{" ".join(str(item) for item in obj)}\n')

        shutil.make_archive('annots', 'zip', annots_dir_path)

        shutil.rmtree(annots_dir_path)
        
        with open('annots.zip', 'rb') as f:
            st.download_button(
                label="Download all annotations",
                data=f,
                file_name="annots.zip",
                mime="application/zip",
            )

        for file_path in preds.keys():
            file_name = Path(file_path).name
            st.image(file_path, caption=file_name)
    else:
        st.error(f'Error while making prediction:\n{prediction_response.json().get('detail')}')

    tmp_dir_deletion_response = requests.get(f'{BASE_URL}/rm_tmp_dir/')