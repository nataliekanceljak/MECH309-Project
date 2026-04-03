# Weather Prediction Using Linear Regression Model

A prediction model for Montreal's weather using linear least squares and time-lag features.

## Description

This project develops a data-driven weather prediction model to forecast temperature and wind speed in Montreal using historical hourly data from the Open-Meteo API. The model is based on linear regression and incorporates current measurements, lagged variables, and periodic time features to capture daily and seasonal patterns. By formulating the problem as a least squares system, the model provides a computationally efficient alternative to large-scale Numerical Weather Prediction (NWP) models while maintaining reasonable predictive accuracy. The model is evaluated using RMSE and MAE and compared against a persistence baseline.

## Getting Started

### Dependencies

The following Python libraries are required with minimum version requirements:
- numpy 1.24
- pandas 2.0
- matplotlib 3.7
- requests 2.31

### Installing

* How/where to download your program
* Any modifications needed to be made to files/folders

### Executing program

* How to run the program
* Step-by-step bullets
```
code blocks for commands
```

## Help
### Common Issues
- #### HTTP 400 Error
  Ensure that the requested dates are in the past (Open-Meteo archive API does not support future dates).
- #### NaN values causing errors
  Make sure .dropna() is applied after creating lag features and target variables.
- #### Plots not showing
  Ensure matplotlib is properly installed and you are running in an environment that supports plotting.

## Authors

Contributors names and contact info

ex. Dominique Pizzie  
ex. [@DomPizzie](https://twitter.com/dompizzie)

## Version History

* 0.2
    * Various bug fixes and optimizations
    * See [commit change]() or See [release history]()
* 0.1
    * Initial Release

## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details

## Acknowledgments

Inspiration, code snippets, etc.
* [awesome-readme](https://github.com/matiassingers/awesome-readme)
* [PurpleBooth](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)
* [dbader](https://github.com/dbader/readme-template)
* [zenorocha](https://gist.github.com/zenorocha/4526327)
* [fvcproductions](https://gist.github.com/fvcproductions/1bfc2d4aecb01a834b46)
