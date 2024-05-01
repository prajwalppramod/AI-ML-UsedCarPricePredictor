from flask import Flask, render_template
import pandas as pd
import os

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


@app.route('/')
def index():
    # Convert DataFrame to HTML table
    table_html = df.to_html(index=False)

    # Render the HTML template with the table data
    return render_template('index.html', table_html=table_html)

if __name__ == '__main__':
    app.run(debug=True)
