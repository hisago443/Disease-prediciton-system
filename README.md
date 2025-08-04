# Disease Prediction System

A machine learning-based web application that predicts diseases based on symptoms using multiple algorithms including Random Forest, SVM, XGBoost, and Logistic Regression.

## 🏥 Features

- **Multi-Model Comparison**: Compare performance of 4 different ML algorithms
- **Interactive Prediction**: Select symptoms and get instant disease predictions
- **Performance Analytics**: Detailed model accuracy analysis and visualizations
- **Real-time Results**: Get predictions from all models simultaneously
- **Beautiful UI**: Modern, responsive web interface built with Streamlit

## 🚀 Live Demo

[Deploy this project on Vercel](#deployment-instructions)

## 📊 Model Performance

The system implements and compares four machine learning models:

| Model | Initial Accuracy | Tuned Accuracy | Improvement |
|-------|------------------|----------------|-------------|
| Random Forest | ~60-70% | 99.90% | +30-40% |
| SVM | ~60-70% | 99.70% | +30-40% |
| XGBoost | ~60-70% | 99.80% | +30-40% |
| Logistic Regression | ~60-70% | 99.90% | +30-40% |

## 🛠️ Technologies Used

- **Python 3.8+**
- **Streamlit** - Web application framework
- **Scikit-learn** - Machine learning library
- **XGBoost** - Gradient boosting framework
- **Pandas & NumPy** - Data manipulation
- **Plotly** - Interactive visualizations
- **Matplotlib & Seaborn** - Static visualizations

## 📁 Project Structure

```
Disease-prediciton-1/
├── app.py                          # Streamlit web application
├── Training.csv                    # Training dataset
├── Testing.csv                     # Testing dataset
├── requirements.txt                # Python dependencies
├── vercel.json                     # Vercel deployment config
├── Procfile                        # Heroku deployment config
└── README.md                      # Project documentation
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. **Clone or download the project**
   ```bash
   git clone <repository-url>
   cd Disease-prediciton-1
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser**
   - The app will automatically open at `http://localhost:8501`
   - If not, manually navigate to the URL shown in the terminal

## 🎯 How to Use

1. **Navigate to "Disease Prediction"** in the sidebar
2. **Select symptoms** you're experiencing by checking the boxes
3. **Click "Predict Disease"** to get instant predictions
4. **View results** from all four models with their accuracies
5. **Explore other sections** for detailed analysis and performance metrics

## 📈 Model Details

### Random Forest
- **Type**: Ensemble learning method
- **Advantages**: Handles non-linear relationships, robust to outliers
- **Use Case**: Complex symptom-disease patterns

### Support Vector Machine (SVM)
- **Type**: Kernel-based classification
- **Advantages**: Effective in high-dimensional spaces
- **Use Case**: Finding optimal decision boundaries

### XGBoost
- **Type**: Gradient boosting algorithm
- **Advantages**: High performance, handles missing values
- **Use Case**: Sequential learning from prediction errors

### Logistic Regression
- **Type**: Linear classification model
- **Advantages**: Interpretable, fast training
- **Use Case**: Baseline model for comparison

## 🚀 Deployment Instructions

### Option 1: Deploy on Vercel (Recommended)

1. **Create a Vercel account** at [vercel.com](https://vercel.com)

2. **Install Vercel CLI**
   ```bash
   npm install -g vercel
   ```

3. **Deploy to Vercel**
   ```bash
   vercel --prod
   ```

4. **Follow the prompts** and get your live URL

5. **Get your live URL** and add it to your CV!

### Option 2: Deploy on Streamlit Cloud (Free)

1. **Push your code to GitHub**

2. **Go to [share.streamlit.io](https://share.streamlit.io)**

3. **Connect your GitHub repository**

4. **Deploy automatically**

5. **Get your live URL**

### Option 3: Deploy on Heroku

1. **Create a `Procfile`**
   ```
   web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```

2. **Create `setup.sh`**
   ```bash
   mkdir -p ~/.streamlit/
   echo "\
   [server]\n\
   headless = true\n\
   port = $PORT\n\
   enableCORS = false\n\
   \n\
   " > ~/.streamlit/config.toml
   ```

3. **Deploy to Heroku**
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

## 📋 For Your CV

### Project Description
```
Disease Prediction System - Machine Learning Web Application
• Built a multi-model ML system using Random Forest, SVM, XGBoost, and Logistic Regression
• Achieved 99.90% accuracy through comprehensive dataset optimization and hyperparameter tuning
• Developed interactive web interface with Streamlit for real-time symptom-based predictions
• Implemented comprehensive analytics dashboard with performance comparisons and visualizations
• Technologies: Python, Streamlit, Scikit-learn, XGBoost, Pandas, Plotly
• Live Demo: [Your deployed URL]
```

### What Happens When Someone Clicks Your Link
1. **Landing Page**: Professional medical-themed interface
2. **Navigation**: Easy-to-use sidebar with multiple sections
3. **Interactive Prediction**: Users can select symptoms and get instant predictions
4. **Model Comparison**: See predictions from all 4 algorithms with accuracies
5. **Analytics**: Detailed performance metrics and visualizations
6. **Professional Presentation**: Clean, modern UI that showcases your skills

## 🔧 Customization

### Adding New Diseases
Edit `create_sample_data.py` to add more diseases and their associated symptoms.

### Modifying Models
Update the model parameters in `app.py` to experiment with different algorithms.

### Styling
Modify the CSS in the `app.py` file to customize the appearance.

## 📝 License

This project is for educational and demonstration purposes. Please note that this should not be used for actual medical diagnosis.

## 🤝 Contributing

Feel free to fork this project and submit pull requests for improvements!

## 📞 Contact

For questions or suggestions, please open an issue in the repository.

---

**Note**: This is a demonstration project and should not be used for actual medical diagnosis. Always consult healthcare professionals for medical advice.
