## Car Price Prediction using Linear Regression

This project demonstrates a simple linear regression model for predicting car prices based on mileage. The project is split into two main scripts: one for training the model and another for making predictions using the trained model. The project also includes functionality for saving and loading model parameters using JSON format.

### Overview

The goal of this project is to create a linear regression model that estimates the price of a car based on its mileage. Linear regression is a foundational machine learning algorithm used to model the relationship between a dependent variable and one or more independent variables.

### Components

1. **Training Script (`train.py`)**:
   - **Purpose**: Trains the linear regression model using a dataset of car mileage and prices.
   - **Features**:
     - Loads data from a CSV file.
     - Normalizes the feature (mileage) for better performance.
     - Implements gradient descent to minimize the cost function and optimize the model parameters (`theta`).
     - Saves the model parameters, normalization mean, and standard deviation to a JSON file for later use.

2. **Prediction Script (`predict.py`)**:
   - **Purpose**: Loads the trained model parameters and uses them to predict the price of a car based on user input for mileage.
   - **Features**:
     - Reads the model parameters and normalization values from the JSON file.
     - Prompts the user to enter the car mileage.
     - Normalizes the input mileage using the saved normalization parameters.
     - Predicts the car price using the trained model and displays the result.

### Files

- `data.csv`: A CSV file containing the dataset with columns for mileage and price. This file is used for training the model.
- `train.py`: Script for training the linear regression model and saving the parameters.
- `predict.py`: Script for predicting car prices using the saved model parameters.
- `model_parameters.json`: JSON file containing the trained model parameters and normalization statistics.

### Setup and Installation

1. **Create a Virtual Environment**:
   - It is recommended to use a virtual environment to manage dependencies. Create one using:
     ```bash
     python -m venv venv
     ```
   - Activate the virtual environment:
     - **On Windows**:
       ```bash
       venv\Scripts\activate
       ```
     - **On macOS/Linux**:
       ```bash
       source venv/bin/activate
       ```

2. **Install Dependencies**:
   - Install the required package using `pip`:
     ```bash
     pip install numpy
     ```

3. **Run the Scripts**:
   - **Training the Model**:
     - Ensure you have your dataset (`data.csv`) in the project directory.
     - Run the training script:
       ```bash
       python train.py
       ```
     - This will generate a `model_parameters.json` file with the model parameters.

   - **Making Predictions**:
     - Ensure you have the `model_parameters.json` file from the training step.
     - Run the prediction script:
       ```bash
       python predict.py
       ```
     - Enter the car mileage when prompted. The script will output the estimated price.

### Requirements

- Python 3.x
- `numpy` library

### Example

**Training Output**:
```
Final theta: [[0.5]
               [10.0]]
```

**Prediction Interaction**:
```
Enter the car mileage: 50000
Predicted price for mileage 50000: $30000.00
```

### Contributing

Feel free to fork this repository and submit pull requests for improvements or bug fixes. Contributions to enhance the model or extend functionality are welcome.