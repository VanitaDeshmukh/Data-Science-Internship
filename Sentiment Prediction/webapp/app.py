from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

model = joblib.load('naive_bayes.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review_text = request.form['review']
    

    prediction = model.predict([review_text])[0]
    sentiment = "Positive" if prediction == 1 else "Negative"
    
    return render_template('result.html', sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
