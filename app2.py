import Flask, request, jsonify, render_template
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the trained model
model = load_model('customer_churn.h5')  # Update with the actual name of your saved model file

@app.route('/')
def home():
    return render_template('index.html')  # Create an HTML file for your home page

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request
        data = request.get_json(force=True)
        input_data = np.array(data['input_data'])

        # Perform prediction using the loaded model
        prediction = model.predict(input_data)
        predicted_class = int(np.round(prediction[0]))

        # Return the result as JSON
        return jsonify({'prediction': predicted_class})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
