import streamlit as st
from pathlib import Path
import os
import shutil
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from backend.main import get_prediction, rm_tmp_dir

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
            {
                'filename': file.name,
                'file': file,
                'filetype': file.type
            }
        )

    prediction_response = get_prediction(
        model_version=model_version,
        files=files
    )
    if prediction_response['status'] == 'failed':
        st.error(prediction_response['details'])
    else:
        preds = prediction_response['results']

        results_dir_path = Path('results')

        try:
            results_dir_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            st.error(f'An error occurred during making directory for predictions. Details:\n{e}')

        try:
            for file_path in preds.keys():
                file_name = Path(file_path).name
                file_stem = Path(file_path).stem
                file_suffix = Path(file_path).suffix

                with open(os.path.join(results_dir_path, file_stem + '.txt'), 'w') as f:
                    f.write('x y x y confidence class_id\n')
                    for obj in preds[file_path]:
                        f.write(f'{" ".join(str(item) for item in obj)}\n')

                shutil.copy(file_path, os.path.join(results_dir_path, 'predicted_image', file_suffix))
        except Exception as e:
            st.error(f'An error occurred during writing predictions results to files. Details:\n{e}')

        try:
            shutil.make_archive('results', 'zip', results_dir_path)
        except Exception as e:
            st.error(f'An error occurred during making predictions archive. Details:\n{e}')

        try:
            shutil.rmtree(results_dir_path)
        except Exception as e:
            st.error(f'An error occurred during removing predictions directory. Details:\n{e}')
        
        try:
            with open('results.zip', 'rb') as f:
                st.download_button(
                    label="Download results",
                    data=f,
                    file_name="results.zip",
                    mime="application/zip",
                )
        except Exception as e:
            st.error(f'An error occurred during opening archive and pressing download button. Details:\n{e}')

        try:
            for file_path in preds.keys():
                file_name = Path(file_path).name
                st.image(file_path, caption=file_name)
        except Exception as e:
            st.error(f'An error occurred during drawing images predictions. Details:\n{e}')

        tmp_dir_deletion_response = rm_tmp_dir()
        if tmp_dir_deletion_response['status'] == 'failed':
            st.error(tmp_dir_deletion_response['details'])