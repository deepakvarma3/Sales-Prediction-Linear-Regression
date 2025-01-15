# Sales-Prediction-Linear-Regression
This project demonstrates the implementation of a Simple Linear Regression model to analyze the relationship between advertising spending and sales for a dietary weight control product.The project explores how advertising impacts sales and provides actionable insights for better decision making in marketing strategies.
The workflow includes:
	1.	Data Loading and Exploration:
	•	The dataset is loaded from an Excel file (Data.xlsx), and statistical exploration is performed to understand the data.
	•	Columns include Advertising (independent variable) and Sales (dependent variable).
	2.	Model Implementation:
	•	Using the scikit-learn library, a linear regression model is trained to predict sales based on advertising spending.
	•	The dataset is split into training and testing sets to evaluate the model’s performance.
	3.	Evaluation and Visualization:
	•	Model performance is assessed using metrics like Mean Squared Error (MSE) and R-squared (R²).
	•	A graph of actual vs. predicted sales is plotted to visualize the model’s accuracy.
	•	A residual plot is generated to analyze the goodness of fit.
	4.	Making Predictions:
	•	The model is used to predict sales for new advertising spending values, demonstrating its practical utility.
	5.	Documentation:
	•	The project includes well-documented code and visualizations to help users understand and replicate the workflow.

Technologies Used:
	•	Python
	•	Pandas (Data Handling)
	•	scikit-learn (Machine Learning)
	•	Matplotlib (Visualization)

Folder Structure:

Sales-Prediction-Linear-Regression/
├── Data.xlsx              # Input dataset
├── main.py                # Main Python script
├── README.md              # Project description
├── requirements.txt       # Required libraries
├── actual_vs_predicted.png # Visualization of results
├── model.pkl              # Saved regression model (optional)

How to Use:
	1.	Clone the repository:

git clone https://github.com/your-username/Sales-Prediction-Linear-Regression.git
cd Sales-Prediction-Linear-Regression


	2.	Install the required dependencies:

pip install -r requirements.txt


	3.	Run the script:

python main.py


	4.	Replace the data in Data.xlsx to use your own dataset.

Features:
	•	Predict future sales based on advertising budget.
	•	Visualize the relationship between advertising and sales.
	•	Easily customizable for other datasets.
