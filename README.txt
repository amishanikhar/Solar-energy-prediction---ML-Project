SOLAR POWER FORECASTING - MINI PROJECT

Files
- Plant_1_Generation_Data.csv, Plant_1_Weather_Sensor_Data.csv : dataset
- train_model.py : cleaning, chronological split, 4-model comparison, saves best model
- app.py         : Flask website (loads saved model/scaler)
- templates/index.html : web UI (responsive: works on phone, tablet, desktop)
- best_solar_model.pkl, scaler.pkl, generation_thresholds.json : trained outputs
- model_comparison.csv, figures/ : results table and report figures
- train_log.txt  : sample terminal output of train_model.py

How to run (terminal, inside this folder)
1. python -m pip install -r requirements.txt
2. python train_model.py     (prints cleaning, split and model comparison; retrains
                              files for YOUR scikit-learn version)
3. python app.py             then open http://127.0.0.1:5000

Key points for viva
- Generation file has two date formats (15-05-2020 and 1/6/2020, both day-first);
  weather file is month-first. Both are parsed explicitly so no rows are lost
  (68,774 of 68,778 rows match after merge).
- SOURCE_KEY (inverter id) is kept so identical readings from different
  inverters (e.g. 0 kW at night) are not wrongly removed as duplicates.
- Split is chronological (first 27 days train, last 7 days test) because
  15-minute readings are highly correlated; a random split leaks information.
- Scaler is fitted on training data only. Predictions are clipped at 0 kW.
- Low/High status cut-offs use the 33rd/67th percentile of daytime power.
- Prediction is AC power per inverter (kW), as in the dataset.
- Best model chosen by lowest RMSE on the unseen test days.
