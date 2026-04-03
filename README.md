# Weather Prediction Using Linear Regression Model

A prediction model for Montreal's weather using linear least squares and time-lag features.

## Description

This project develops a data-driven weather prediction model to forecast temperature and wind speed in Montreal using historical hourly data from the Open-Meteo API. The model is based on linear regression and incorporates current measurements, lagged variables, and periodic time features to capture daily and seasonal patterns. By formulating the problem as a least squares system, the model provides a computationally efficient alternative to large-scale Numerical Weather Prediction (NWP) models while maintaining reasonable predictive accuracy. The model is evaluated using RMSE and MAE and compared against a persistence baseline.

This project is part of MECH 309 Numerical Methods at McGill University located in Montreal, QC.

## Getting Started

### Dependencies

The following Python libraries are required (minimum versions):
- numpy >= 1.24  
- pandas >= 2.0  
- matplotlib >= 3.7  
- requests >= 2.31  

### Installing

1. Clone the repository or download as a zip file.
2. Navigate to the project folder.

### Executing program

1. Open the file in Visual Studio Code or another IDE.  
2. Set the `start_date` and `end_date` for the analysis period.  
3. Adjust `val_hours` to modify the validation (prediction) interval.  
4. Run the Python script.

The program will output:

- Optimized feature sets for each prediction horizon  
- RMSE and MAE values for temperature, wind speed, and baseline models  
- Plots comparing predicted vs actual values (including baseline)  
- Standard deviation of temperature and wind for summer and winter periods

## Help
### Common Issues
- **HTTP 400 Error**  
  Ensure that the requested dates are in the past (the Open-Meteo archive API does not support future dates).

- **NaN values causing errors**  
  Ensure `.dropna()` is applied after creating lag features and target variables.

- **Plots not displaying**  
  Ensure matplotlib is installed and your environment supports plotting.

## Authors

Sally Jeon — Mechanical Engineering Student, McGill University
Natalie Kanceljak — Mechanical Engineering Student, McGill University

## Acknowledgments

We acknowledge Professor Nikolai in his guidance with creating this model and providing the starter code that served as a basis of our prediction model. We also acknowledge Open-Meteo API for providing historical weather data.
