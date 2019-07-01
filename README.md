# Linear-Regression-with-Gradient-Descent

## Predict fuel consumption in Miles per Gallon (mpg)

For this purpose, I'm using UCI [MPG dataset](https://archive.ics.uci.edu/ml/datasets/auto+mpg) which has these features.
Columns:
    1. mpg:           continuous
    2. cylinders:     multi-valued discrete
    3. displacement:  continuous
    4. horsepower:    continuous
    5. weight:        continuous
    6. acceleration:  continuous
    7. model year:    multi-valued discrete
    8. origin:        multi-valued discrete
    9. car name:      string (unique for each instance)


## How to run this project
Go to the project directory and build docker image using this command.

```
docker build -t linear .
```

Run docker container:
```
docker run --name linear_container -p5000:5000 linear:latest
```

Once container starts running, you can access your application using these endpoints:

### To train the model: http://0.0.0.0:5000/train


### To test the model, post data to get prediction

URL: http://0.0.0.0:5000/test
```json
{ "cylinders": 8, "displacement": 307.0, "horsepower": 130.0, "weight": 3504, "acceleration": 12.0, "model_year": 70, "origin": 1}
```


Note: You can run **code.py** locally to train model using custom implementation of Linear Regression, Gradient Descent, Root Mean Squared Error, R Square score and compare the accuracy against sklearn implementation of Linear Regression. 