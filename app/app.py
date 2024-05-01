from flask import Flask, render_template, request
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Get the directory path of the current script (app.py)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute path to the CSV file
csv_file_path = os.path.join(current_dir, '..', 'data', 'cars_data.csv')

# Read the CSV file
df = pd.read_csv(csv_file_path)

# Drop rows where 'Power' is 0
df = df[df['Power'] != 0]

# Drop the entire 'New_Price' column
df.drop(columns=['New_Price'], inplace=True)

# Remove rows with NaN in any column or 'null bhp' in the Power column
df = df.dropna()  # Drop rows with any NaN values in any column
df = df[df['Power'] != 'null bhp']  # Drop rows with 'null bhp' in the Power column

# Preprocess the data
label_encoders = {}
for column in ['Fuel_Type', 'Transmission', 'Owner_Type']:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

# Train a linear regression model
X = df[['Year', 'Kilometers_Driven', 'Fuel_Type', 'Transmission', 'Owner_Type', 'Seats']]
y = df['Price']
model = LinearRegression()
model.fit(X, y)

# Define route for index page
@app.route('/')
def index():
    return render_template('index.html')

# Define route for form submission
@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    year = int(request.form['year'])
    km_driven = int(request.form['km_driven'])
    fuel_type = label_encoders['Fuel_Type'].transform([request.form['fuel_type']])[0]
    transmission = label_encoders['Transmission'].transform([request.form['transmission']])[0]
    owner_type = label_encoders['Owner_Type'].transform([request.form['owner_type']])[0]
    seats = int(request.form['seats'])
    
    # Predict the price
    predicted_price = model.predict([[year, km_driven, fuel_type, transmission, owner_type, seats]])
    
    # Print the predicted price for debugging
    print("Predicted Price:", predicted_price)
    
    # Convert the predicted price to a string
    predicted_price_str = "{:.2f}".format(predicted_price[0])
    
    return render_template('result.html', price=predicted_price_str)  # Change predicted_price to price


if __name__ == '__main__':
    app.run(debug=True)
