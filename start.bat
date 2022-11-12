cd data
python process_data.py disaster_messages.csv disaster_categories.csv DR.db
cd ../
cd models
python train_classifier.py ../data/DR.db cf.pkl
cd ../
cd app
python run.py