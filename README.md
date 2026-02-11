# ML hishing Detector System

Doble layer system (Random Forest + Logistic Regression) for detect phishing on electronic emails.


## Instalation

```bash
pip install -r requirements.txt
cd src
python 01_download_data.py
python 02_preprocess.py
python 03_train_models.py
python 04_detector.py
