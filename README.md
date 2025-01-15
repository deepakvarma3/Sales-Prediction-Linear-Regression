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
[Uploading Copy_of_Dietary_weight_control_product.ipynb…](){
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "jYAEyeBkS7iL",
        "outputId": "0dd582a9-3bc9-40f4-e148-822ee926f361"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: pandas in /usr/local/lib/python3.10/dist-packages (2.2.2)\n",
            "Requirement already satisfied: openpyxl in /usr/local/lib/python3.10/dist-packages (3.1.5)\n",
            "Requirement already satisfied: scikit-learn in /usr/local/lib/python3.10/dist-packages (1.6.0)\n",
            "Requirement already satisfied: matplotlib in /usr/local/lib/python3.10/dist-packages (3.8.0)\n",
            "Requirement already satisfied: numpy>=1.22.4 in /usr/local/lib/python3.10/dist-packages (from pandas) (1.26.4)\n",
            "Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.10/dist-packages (from pandas) (2.8.2)\n",
            "Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas) (2024.2)\n",
            "Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.10/dist-packages (from pandas) (2024.2)\n",
            "Requirement already satisfied: et-xmlfile in /usr/local/lib/python3.10/dist-packages (from openpyxl) (2.0.0)\n",
            "Requirement already satisfied: scipy>=1.6.0 in /usr/local/lib/python3.10/dist-packages (from scikit-learn) (1.13.1)\n",
            "Requirement already satisfied: joblib>=1.2.0 in /usr/local/lib/python3.10/dist-packages (from scikit-learn) (1.4.2)\n",
            "Requirement already satisfied: threadpoolctl>=3.1.0 in /usr/local/lib/python3.10/dist-packages (from scikit-learn) (3.5.0)\n",
            "Requirement already satisfied: contourpy>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib) (1.3.1)\n",
            "Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.10/dist-packages (from matplotlib) (0.12.1)\n",
            "Requirement already satisfied: fonttools>=4.22.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib) (4.55.3)\n",
            "Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib) (1.4.7)\n",
            "Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib) (24.2)\n",
            "Requirement already satisfied: pillow>=6.2.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib) (11.0.0)\n",
            "Requirement already satisfied: pyparsing>=2.3.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib) (3.2.0)\n",
            "Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil>=2.8.2->pandas) (1.17.0)\n"
          ]
        }
      ],
      "source": [
        "pip install  pandas openpyxl scikit-learn matplotlib"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.linear_model import LinearRegression\n",
        "from sklearn.metrics import mean_squared_error, r2_score\n",
        "import matplotlib.pyplot as plt"
      ],
      "metadata": {
        "id": "SvZlpukNVwUp"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Load the dataset\n",
        "data = pd.read_excel('Data.xlsx')\n",
        "# Preview the data\n",
        "print(data.head())"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "T2Imda66WLGh",
        "outputId": "20525a97-f984-4821-a03e-3f954b761496"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "     12  15\n",
            "0  20.5  16\n",
            "1  21.0  18\n",
            "2  15.5  27\n",
            "3  15.3  21\n",
            "4  23.5  49\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Get dataset info\n",
        "print(\"\\nDataset Info:\")\n",
        "print(data.info())\n",
        "# Get summary statistics\n",
        "print(\"\\nSummary Statistics:\")\n",
        "print(data.describe())"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "cg7IfKTmWgpE",
        "outputId": "b7a077f7-8a4c-480b-acef-6c3814526640"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Dataset Info:\n",
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 35 entries, 0 to 34\n",
            "Data columns (total 2 columns):\n",
            " #   Column  Non-Null Count  Dtype  \n",
            "---  ------  --------------  -----  \n",
            " 0   12      35 non-null     float64\n",
            " 1   15      35 non-null     int64  \n",
            "dtypes: float64(1), int64(1)\n",
            "memory usage: 688.0 bytes\n",
            "None\n",
            "\n",
            "Summary Statistics:\n",
            "              12         15\n",
            "count  35.000000  35.000000\n",
            "mean   24.605714  28.914286\n",
            "std     5.902290  18.905915\n",
            "min    15.300000   1.000000\n",
            "25%    20.600000  16.500000\n",
            "50%    24.500000  24.000000\n",
            "75%    28.900000  42.000000\n",
            "max    36.500000  65.000000\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(data.columns)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "6_droDw0Xfw5",
        "outputId": "5b60856b-f42c-447d-99e3-b456a35cbdd6"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Index([12, 15], dtype='int64')\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "x = data[[12]]  # Access column 12 as an integer\n",
        "y = data[15]    # Access column 15 as an integer"
      ],
      "metadata": {
        "id": "3yRmYJQdY1ND"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Spli the datasets into training and testing sets\n",
        "x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)"
      ],
      "metadata": {
        "id": "mtghRho7ZrJ6"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Initialize and train the model\n",
        "model = LinearRegression()\n",
        "model.fit(x_train, y_train)\n",
        "# Print the model coefficients\n",
        "print(\"\\nModel Coefficients:\")\n",
        "print(f\"Intercept: {model.intercept_}\")\n",
        "print(f\"Coefficient: {model.coef_[0]}\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "7pO5cQWibWOP",
        "outputId": "c02658f4-206c-451f-9f3f-7d33035ee1bb"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Model Coefficients:\n",
            "Intercept: -22.181763451269877\n",
            "Coefficient: 2.036016276149995\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Predict on the test set\n",
        "y_pred = model.predict(x_test)"
      ],
      "metadata": {
        "id": "8IXdvwHXcaVY"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Calculate metrics\n",
        "mse = mean_squared_error(y_test, y_pred)\n",
        "r2 = r2_score(y_test, y_pred)\n",
        "print(\"\\nMetrics:\")\n",
        "print(f\"Mean Squared Error (MSE): {mse}\")\n",
        "print(f\"R-squared (R2): {r2}\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "3pdUHu8Kcpm7",
        "outputId": "6787cd92-448f-482b-8bde-92691f2aeebc"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Metrics:\n",
            "Mean Squared Error (MSE): 224.09629706236424\n",
            "R-squared (R2): 0.15792035613068645\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Scatter plot for actual vs predicted\n",
        "plt.scatter(x_test, y_test, color='blue', label='Actual Sales')\n",
        "plt.plot(x_test, y_pred, color='red', label='Regression Line')\n",
        "plt.xlabel('Advertising')\n",
        "plt.ylabel('Sales')\n",
        "plt.title('Actual vs Predicted Sales')\n",
        "plt.legend()\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 472
        },
        "id": "V58dh6VjdJx-",
        "outputId": "174bf4b5-36a9-46d1-c36e-7c24dfafbb27"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "plt.savefig('actual_vs_predicted.png')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        },
        "id": "uxMrs2MdeTgv",
        "outputId": "17309e0f-258c-422f-f754-c661381ca0aa"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 0 Axes>"
            ]
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Evaluate the model\n",
        "mse = mean_squared_error(y_test, y_pred)\n",
        "r2 = r2_score(y_test, y_pred)\n",
        "print(f\"Mean Squared Error (MSE): {mse}\")\n",
        "print(f\"R-squared (R2): {r2}\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "2xtiEaare1-i",
        "outputId": "08a6a50d-0c25-4d40-9083-4f2970870435"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Mean Squared Error (MSE): 224.09629706236424\n",
            "R-squared (R2): 0.15792035613068645\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Calculate residuals\n",
        "residuals = y_test - y_pred\n",
        "# Plot residuals\n",
        "plt.scatter(y_pred, residuals, color='purple')\n",
        "plt.axhline(y=0, color='black', linestyle='--')\n",
        "plt.xlabel('Predicted Sales')\n",
        "plt.ylabel('Residuals')\n",
        "plt.title('Residual Plot')\n",
        "plt.show()\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 472
        },
        "id": "4MZkHuQLfIja",
        "outputId": "82fb0942-9a0a-4ab7-ea0a-043919f3f1b4"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAj4AAAHHCAYAAAC/R1LgAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjguMCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy81sbWrAAAACXBIWXMAAA9hAAAPYQGoP6dpAAA3rUlEQVR4nO3dfVhUdf7/8dcAgigMiIKogGiawnrX6uZieVOaYq1pSLjdapvbqljel+5vS2sryq1V20zb2tSt7VvelrlG+TUlt8xS10yXSF0TWkGx1hm8AZU5vz+s+TZyI+LAAJ/n47rmujif854z7zkez7zmzJkzNsuyLAEAABjAz9cNAAAA1BaCDwAAMAbBBwAAGIPgAwAAjEHwAQAAxiD4AAAAYxB8AACAMQg+AADAGAQfAABgDIIPgDpnzpw5stlsVaq12WyaM2dOjfYzYMAADRgwoM4uD0DVEXwAVGjp0qWy2WzuW0BAgNq0aaMxY8boP//5j6/bq3Pi4+M91ldUVJT69u2rNWvWeGX5p06d0pw5c7R582avLA8wEcEHwEU99thjevXVV7V48WINHTpUr732mvr376/i4uIaebzf/e53On36dI0su6b16NFDr776ql599VVNnz5dhw8fVkpKihYvXnzZyz516pQeffRRgg9wGQJ83QCAum/o0KHq1auXJGns2LFq0aKFnn76aa1du1ZpaWlef7yAgAAFBNTP3VObNm105513uqfvvvtudejQQfPmzdO4ceN82BkAiSM+AKqhb9++kqQDBw54jH/55ZdKTU1VRESEGjdurF69emnt2rUeNWfPntWjjz6qjh07qnHjxmrevLmuvfZabdiwwV1T3jk+JSUlmjJliiIjIxUaGqqbb75Z33zzTZnexowZo/j4+DLj5S1zyZIluv766xUVFaWgoCAlJiZq0aJFl7QuLiY6OloJCQk6ePBgpXVHjx7Vvffeq5YtW6px48bq3r27li1b5p7/9ddfKzIyUpL06KOPuj9Oq+nzm4CGpn6+pQLgU19//bUkqVmzZu6xvXv36pprrlGbNm00c+ZMNW3aVMuXL9eIESO0atUq3XLLLZLOB5CMjAyNHTtWV199tZxOp7Zv366dO3fqhhtuqPAxx44dq9dee0233367+vTpow8++EA33XTTZT2PRYsW6Sc/+YluvvlmBQQE6J133tGECRPk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B2vQoEHat29fjfUDAADql2oFn5CQEH377beSpPfff1833HCDJKlx48Y6ffq097q7wMmTJ9W9e3ctXLiw3Plz587Vc889p8WLF2vbtm1q2rSphgwZouLi4hrrCQAA1B/V+smKG264QWPHjtVVV13lcbXmvXv3Kj4+3pv9eRg6dKiGDh1a7jzLsjR//nz97ne/0/DhwyVJf/3rX9WyZUu99dZb+uUvf1ljfQEAgPqhWkd8Fi5cqKSkJBUWFmrVqlVq3ry5JGnHjh267bbbvNpgVR08eFAFBQUaNGiQeywsLEy9e/fW1q1bK7xfSUmJnE6nxw0AADRM1TriEx4erueff77M+KOPPnrZDVVXQUGBJKlly5Ye4y1btnTPK09GRoZP+wYAALWnWkd8JGnLli2688471adPH/3nP/+RJL366qv6xz/+4bXmasOsWbPkcDjct7y8PF+3BAAAaki1gs+qVas0ZMgQBQcHa+fOnSopKZEkORwOPfnkk15tsKqio6MlSUeOHPEYP3LkiHteeYKCgmS32z1uAACgYapW8Hn88ce1ePFivfTSS2rUqJF7/JprrtHOnTu91tylaNeunaKjo7Vx40b3mNPp1LZt25SUlOSTngAAQN1SrXN8cnJy1K9fvzLjYWFhOn78+OX2VKETJ05o//797umDBw9q165dioiIUFxcnCZPnqzHH39cHTt2VLt27fTwww+rdevWGjFiRI31BAAA6o9qBZ/o6Gjt37+/zFfX//GPf6h9+/be6Ktc27dv13XXXeeenjp1qiRp9OjRWrp0qR588EGdPHlS9913n44fP65rr71WmZmZaty4cY31BAAA6g+bZVnWpd4pIyNDr732ml555RXdcMMNWr9+vQ4dOqTJkyfrkUce0f33318TvdYKp9OpsLAwORwOzvcBAKCeqOrrd7WO+MycOVMul0sDBw7UqVOn1K9fPwUFBWnGjBkaO3ZstZsGAACoSdU6udlms+n//b//p++++0579uzRJ598osLCQoWFhaldu3be7hEAAMArLin4lJSUaNasWerVq5euueYarV+/XomJidq7d686deqkBQsW8OvsAACgzrqkj7oeeeQRvfjiixo0aJA+/vhj3Xrrrbrnnnv0ySef6Nlnn9Wtt94qf3//muoVAADgslxS8FmxYoX++te/6uabb9aePXvUrVs3nTt3Tp9//rlsNltN9QgAAOAVl/RR1zfffKOePXtKkrp06aKgoCBNmTKF0AMAAOqFSwo+paWlCgwMdE8HBAQoJCTE600BAADUhEv6qMuyLI0ZM0ZBQUGSpOLiYo0bN05Nmzb1qFu9erX3OgQAAPCSSwo+o0eP9pi+8847vdoMAABATbqk4LNkyZKa6gMAAKDGVesChgAAAPURwQcAABiD4AMAAIxB8AEAAMYg+AAAAGMQfAAAgDEIPgAAwBgEHwAAYAyCDwAAMMYlXbkZ1eMqdSl3S66K8osU2ipUcX3j5OdP5gQAoLYRfGpY9upsZU7KlPMbp3vMHmNX8oJkJaQk+LAzAADMw2GHGpS9OlvLU5d7hB5Jcv7HqeWpy5W9OttHnQEAYCaCTw1xlbqUOSlTssqZ+f1Y5uRMuUpdtdoXAAAmI/jUkNwtuWWO9HiwJGeeU7lbcmuvKQAADEfwqSFF+UVerQMAAJeP4FNDQluFerUOAABcPoJPDYnrGyd7jF2yVVBgk+yxdsX1javVvgAAMBnBp4b4+fspeUHy+YkLw8/308nzk7meDwAAtYhX3RqUkJKgtJVpsrexe4zbY+xKW5nGdXwAAKhlXMCwhiWkJKjT8E5cuRkAgDqA4FML/Pz9FD8g3tdtAABgPA47AAAAYxB8AACAMQg+AADAGAQfAABgDIIPAAAwBsEHAAAYg+ADAACMQfABAADGIPgAAABjEHwAAIAxCD4AAMAYBB8AAGAMgg8AADAGwQcAABiD4AMAAIxB8AEAAMYg+AAAAGMQfAAAgDEIPgAAwBgEHwAAYAyCDwAAMAbBBwAAGIPgAwAAjEHwAQAAxiD4AAAAYxB8AACAMQg+AADAGAQfAABgDIIPAAAwBsEHAAAYg+ADAACMQfABAADGIPgAAABjEHwAAIAxCD4AAMAYBB8AAGAMgg8AADAGwQcAABijQQafhQsXKj4+Xo0bN1bv3r316aef+rolAKgzXKUufb35a33xP1/o681fy1Xq8nVLQK0J8HUD3vbmm29q6tSpWrx4sXr37q358+dryJAhysnJUVRUlK/bAwCfyl6drcxJmXJ+43SP2WPsSl6QrISUBB92BtQOm2VZlq+b8KbevXvrZz/7mZ5//nlJksvlUmxsrO6//37NnDnzovd3Op0KCwvT4cOHZbfby8z39/dX48aN3dMnT56scFl+fn4KDg6uVu2pU6dU0T+NzWZTkyZNqlV7+vRpuVwVv7tr2rRptWqLi4tVWlrqldomTZrIZrNJkkpKSnTu3Dmv1AYHB8vP7/xBzjNnzujs2bNeqW3cuLH8/f0vufbs2bM6c+ZMhbVBQUEKCAi45Npz586ppKSkwtrAwEA1atTokmtLS0tVXFxcYW2jRo0UGBh4ybUul0unT5/2Sm1AQICCgoIkSZZl6dSpU16pvZT/93V5H/Hl219q9R2ry9QH2s6v37SVaYofGs8+Quwj6uM+4ofXb4fDUe7rt5vVgJSUlFj+/v7WmjVrPMbvvvtu6+abby73PsXFxZbD4XDf8vLyLEkV3m688UaP+zdp0qTC2v79+3vUtmjRosLaXr16edS2bdu2wtrExESP2sTExApr27Zt61Hbq1evCmtbtGjhUdu/f/8Ka5s0aeJRe+ONN1a63n4sNTW10toTJ064a0ePHl1p7dGjR921EyZMqLT24MGD7trp06dXWrtnzx537ezZsyut/fTTT921c+fOrbR206ZN7trnn3++0tp169a5a5csWVJp7fLly921y5cvr7R2yZIl7tp169ZVWvv888+7azdt2lRp7dy5c921n376aaW1s2fPdtfu2bOn0trp06e7aw8ePFhp7YQJE9y1R48erbR29OjR7toTJ05UWpuamuqxDVdWW9/2EWEKs+ZojjXHNsf6Y+wf2Ud8j33EefVpH+FwOCxJlsPhsCrToM7xOXbsmEpLS9WyZUuP8ZYtW6qgoKDc+2RkZCgsLMx9i42NrY1WAaBusSRnnlNniio+agA0BA3qo67Dhw+rTZs2+vjjj5WUlOQef/DBB5WVlaVt27aVuU9JSYnHYTyn06nY2Fg+6rrEWg5jcxibj7ouvbY29xFfvPmF1v5qbbn1gQp0/33j0huVmJpYYR/sIy69ln3EeXXlo64GdXJzixYt5O/vryNHjniMHzlyRNHR0eXeJygoyL3z+7GmTZt6/EesSFVqqlP7452WN2t/vOP0Zu2Pd/TerK3o3+dyawMDA93/qXxV26hRI/cOw5u1AQEB7h2cN2v9/f2rvA1fSq2fn1+N1NpsthqplWru/31N7iNatmvpEXAqEtk2ssp9sI+o2Vr2Eeddyv/7Ki3Pa0uqAwIDA9WzZ09t3LjRPeZyubRx40aPI0AAYJq4vnGyx9glWwUFNskea1dc37ha7QuobQ0q+EjS1KlT9dJLL2nZsmXKzs7W+PHjdfLkSd1zzz2+bg0AfMbP30/JC5LPT1wYfr6fTp6fLD//BveyAHhoUB91SdKoUaNUWFioRx55RAUFBerRo4cyMzPLnPAMAKZJSElQ2sq08q/jM5/r+MAMDerkZm+o8nUAAKCecpW6lLslV0X5RQptFaq4vnEc6UG9Z+TJzQCAi/Pz91P8gHhftwH4BBEfAAAYg+ADAACMQfABAADGIPgAAABjEHwAAIAxCD4AAMAYBB8AAGAMgg8AADAGwQcAABiD4AMAAIxB8AEAAMYg+AAAAGMQfAAAgDEIPgAAwBgEHwAAYAyCDwAAMAbBBwAAGIPgAwAAjEHwAQAAxiD4AAAAYxB8AACAMQg+AADAGAQfAABgDIIPAAAwBsEHAAAYg+ADAACMQfABAADGIPgAAABjEHwAAIAxCD4AAMAYBB8AAGAMgg8AADAGwQcAABiD4AMAAIxB8AEAAMYI8HUD8C5XqUu5W3JVlF+k0FahiusbJz9/8i0AABLBp0HJXp2tzEmZcn7jdI/ZY+xKXpCshJQEH3YGADBdXXljTvBpILJXZ2t56nLJ8hx3/sep5anLlbYyjfADAPCJuvTGnM9AGgBXqUuZkzLLhB5J7rHMyZlylbpqtS8AAH54Y/7j0CP93xvz7NXZtdoPwacByN2SW2aD8mBJzjyncrfk1l5TAADj1cU35gSfBqAov8irdQAAeENdfGNO8GkAQluFerUOAABvqItvzAk+DUBc3zjZY+ySrYICm2SPtSuub1yt9gUAMFtdfGNO8GkA/Pz9lLwg+fzEheHn++nk+clczwcAUKvq4htzXgkbiISUBKWtTJO9jd1j3B5j56vsAACfqItvzG2WZZV3rrWxnE6nwsLC5HA4ZLfbL36HOqauXCAKAIAflHsdn1i7kud77zo+VX39JvhcoL4HHwAA6qKafmNe1ddvrtwMAABqnJ+/n+IHxPu6Dc7xAQAA5iD4AAAAYxB8AACAMQg+AADAGAQfAABgDIIPAAAwBsEHAAAYg+ADAACMQfABAADG4MrNAPiNNwDGIPgAhiv3xwNj7Epe4L0fDwSAuoK3dIDBsldna3nqco/QI0nO/zi1PHW5sldn+6gzAKgZBB/AUK5SlzInZUpWOTO/H8ucnClXqatW+wKAmkTwAQyVuyW3zJEeD5bkzHMqd0tu7TUFADWM4AMYqii/yKt1AFAfEHwAQ4W2CvVqHQDUBwQfwFBxfeNkj7FLtgoKbJI91q64vnG12hcA1CSCD2AoP38/JS9IPj9xYfj5fjp5fjLX8wHQoLBHAwyWkJKgtJVpsrexe4zbY+xKW5nGdXwANDhcwBAwXEJKgjoN78SVmwEYod7s2Z544gn16dNHTZo0UXh4eLk1ubm5uummm9SkSRNFRUVpxowZOnfuXO02CtRDfv5+ih8Qr663dVX8gHhCD4AGq94c8Tlz5oxuvfVWJSUl6S9/+UuZ+aWlpbrpppsUHR2tjz/+WPn5+br77rvVqFEjPfnkkz7oGAAA1DU2y7LKu25rnbV06VJNnjxZx48f9xh/99139Ytf/EKHDx9Wy5YtJUmLFy/WQw89pMLCQgUGBlZp+U6nU2FhYXI4HLLb7Re/AwAA8Lmqvn43mOPZW7duVdeuXd2hR5KGDBkip9OpvXv3Vni/kpISOZ1OjxsAAGiYGkzwKSgo8Ag9ktzTBQUFFd4vIyNDYWFh7ltsbGyN9gkAAHzHp8Fn5syZstlsld6+/PLLGu1h1qxZcjgc7lteXl6NPh4AAPAdn57cPG3aNI0ZM6bSmvbt21dpWdHR0fr00089xo4cOeKeV5GgoCAFBQVV6TEAAED95tPgExkZqcjISK8sKykpSU888YSOHj2qqKgoSdKGDRtkt9uVmJjolccAAAD1W735Ontubq6+++475ebmqrS0VLt27ZIkdejQQSEhIRo8eLASExN11113ae7cuSooKNDvfvc7paenc0QHAABIqkdfZx8zZoyWLVtWZnzTpk0aMGCAJOnQoUMaP368Nm/erKZNm2r06NF66qmnFBBQ9XzH19kBAKh/qvr6XW+CT21xOBwKDw9XXl4ewQcAgHrC6XQqNjZWx48fV1hYWIV19eajrtpSVFQkSXytHQCAeqioqKjS4MMRnwu4XC4dPnxYoaGhstlsvm6nWn5IvaYftWI9nMd6OI/1cB7r4TzWw3kNaT1YlqWioiK1bt1afn4VX62HIz4X8PPzU0xMjK/b8Aq73V7vN2RvYD2cx3o4j/VwHuvhPNbDeQ1lPVR2pOcHDebKzQAAABdD8AEAAMYg+DRAQUFBmj17tvHXL2I9nMd6OI/1cB7r4TzWw3kmrgdObgYAAMbgiA8AADAGwQcAABiD4AMAAIxB8AEAAMYg+NRjGRkZ+tnPfqbQ0FBFRUVpxIgRysnJ8agpLi5Wenq6mjdvrpCQEI0cOVJHjhzxUcc1oyrrYcCAAbLZbB63cePG+ajjmrFo0SJ169bNfSGypKQkvfvuu+75JmwLF1sHJmwH5Xnqqadks9k0efJk95gJ28OFylsPJmwTc+bMKfMcO3fu7J5v2rZA8KnHsrKylJ6erk8++UQbNmzQ2bNnNXjwYJ08edJdM2XKFL3zzjtasWKFsrKydPjwYaWkpPiwa++rynqQpF//+tfKz8933+bOneujjmtGTEyMnnrqKe3YsUPbt2/X9ddfr+HDh2vv3r2SzNgWLrYOpIa/HVzos88+04svvqhu3bp5jJuwPfxYRetBMmOb+MlPfuLxHP/xj3+455m2LchCg3H06FFLkpWVlWVZlmUdP37catSokbVixQp3TXZ2tiXJ2rp1q6/arHEXrgfLsqz+/ftbkyZN8l1TPtKsWTPr5ZdfNnZbsKz/WweWZd52UFRUZHXs2NHasGGDx3M3bXuoaD1YlhnbxOzZs63u3buXO8+0bcGyLIsjPg2Iw+GQJEVEREiSduzYobNnz2rQoEHums6dOysuLk5bt271SY+14cL18IO//e1vatGihbp06aJZs2bp1KlTvmivVpSWluqNN97QyZMnlZSUZOS2cOE6+IFJ20F6erpuuukmj393ybx9Q0Xr4QcmbBP79u1T69at1b59e91xxx3Kzc2VZN62IPEjpQ2Gy+XS5MmTdc0116hLly6SpIKCAgUGBio8PNyjtmXLliooKPBBlzWvvPUgSbfffrvatm2r1q1ba/fu3XrooYeUk5Oj1atX+7Bb7/viiy+UlJSk4uJihYSEaM2aNUpMTNSuXbuM2RYqWgeSOduBJL3xxhvauXOnPvvsszLzTNo3VLYeJDO2id69e2vp0qXq1KmT8vPz9eijj6pv377as2ePUdvCDwg+DUR6err27Nnj8bmtiSpaD/fdd5/7765du6pVq1YaOHCgDhw4oCuuuKK226wxnTp10q5du+RwOLRy5UqNHj1aWVlZvm6rVlW0DhITE43ZDvLy8jRp0iRt2LBBjRs39nU7PlOV9WDCNjF06FD33926dVPv3r3Vtm1bLV++XMHBwT7szDf4qKsBmDhxotatW6dNmzYpJibGPR4dHa0zZ87o+PHjHvVHjhxRdHR0LXdZ8ypaD+Xp3bu3JGn//v210VqtCQwMVIcOHdSzZ09lZGSoe/fuWrBggVHbQkXroDwNdTvYsWOHjh49qp/+9KcKCAhQQECAsrKy9NxzzykgIEAtW7Y0Ynu42HooLS0tc5+Guk38WHh4uK688krt37/fqH3DDwg+9ZhlWZo4caLWrFmjDz74QO3atfOY37NnTzVq1EgbN250j+Xk5Cg3N9fjnIf67mLroTy7du2SJLVq1aqGu/Mtl8ulkpISY7aF8vywDsrTULeDgQMH6osvvtCuXbvct169eumOO+5w/23C9nCx9eDv71/mPg11m/ixEydO6MCBA2rVqpWZ+wZfn12N6hs/frwVFhZmbd682crPz3ffTp065a4ZN26cFRcXZ33wwQfW9u3braSkJCspKcmHXXvfxdbD/v37rccee8zavn27dfDgQevtt9+22rdvb/Xr18/HnXvXzJkzraysLOvgwYPW7t27rZkzZ1o2m816//33LcsyY1uobB2Ysh1U5MJvL5mwPZTnx+vBlG1i2rRp1ubNm62DBw9aH330kTVo0CCrRYsW1tGjRy3LMm9bIPjUY5LKvS1ZssRdc/r0aWvChAlWs2bNrCZNmli33HKLlZ+f77uma8DF1kNubq7Vr18/KyIiwgoKCrI6dOhgzZgxw3I4HL5t3Mt+9atfWW3btrUCAwOtyMhIa+DAge7QY1lmbAuVrQNTtoOKXBh8TNgeyvPj9WDKNjFq1CirVatWVmBgoNWmTRtr1KhR1v79+93zTdsWbJZlWb451gQAAFC7OMcHAAAYg+ADAACMQfABAADGIPgAAABjEHwAAIAxCD4AAMAYBB8AAGAMgg+AOmHMmDEaMWKEe3rAgAGaPHlyrfexefNm2Wy2Mr9d5G0XPl8AtYPgA6BCY8aMkc1mk81mc//452OPPaZz587V+GOvXr1av//976tUW1th5Qeff/65br75ZkVFRalx48aKj4/XqFGjdPTo0Vp5fADVR/ABUKnk5GTl5+dr3759mjZtmubMmaM//OEP5daeOXPGa48bERGh0NBQry3PWwoLCzVw4EBFRETovffeU3Z2tpYsWaLWrVvr5MmTvm4PwEUQfABUKigoSNHR0Wrbtq3Gjx+vQYMGae3atZL+7+OaJ554Qq1bt1anTp0kSXl5eUpLS1N4eLgiIiI0fPhwff311+5llpaWaurUqQoPD1fz5s314IMP6sJfz7nwo66SkhI99NBDio2NVVBQkDp06KC//OUv+vrrr3XddddJkpo1ayabzaYxY8ZIOv/L7BkZGWrXrp2Cg4PVvXt3rVy50uNx1q9fryuvvFLBwcG67rrrPPosz0cffSSHw6GXX35ZV111ldq1a6frrrtO8+bNU7t27dzP795773U/bqdOnbRgwYJKl3uxXv/73//qjjvuUGRkpIKDg9WxY0ctWbKk0mUCKCvA1w0AqF+Cg4P17bffuqc3btwou92uDRs2SJLOnj2rIUOGKCkpSVu2bFFAQIAef/xxJScna/fu3QoMDNSzzz6rpUuX6pVXXlFCQoKeffZZrVmzRtdff32Fj3v33Xdr69ateu6559S9e3cdPHhQx44dU2xsrFatWqWRI0cqJydHdrtdwcHBkqSMjAy99tprWrx4sTp27KgPP/xQd955pyIjI9W/f3/l5eUpJSVF6enpuu+++7R9+3ZNmzat0ucfHR2tc+fOac2aNUpNTZXNZitT43K5FBMToxUrVqh58+b6+OOPdd9996lVq1ZKS0srd7kX6/Xhhx/Wv/71L7377rtq0aKF9u/fr9OnT1/03wvABXz8I6kA6rDRo0dbw4cPtyzLslwul7VhwwYrKCjImj59unt+y5YtrZKSEvd9Xn31VatTp06Wy+Vyj5WUlFjBwcHWe++9Z1mWZbVq1cqaO3eue/7Zs2etmJgY92NZluevaOfk5FiSrA0bNpTb56ZNmyxJ1n//+1/3WHFxsdWkSRPr448/9qi99957rdtuu82yLMuaNWuWlZiY6DH/oYceKrOsC/32t7+1AgICrIiICCs5OdmaO3euVVBQUGG9ZVlWenq6NXLkSPf0j9dtVXodNmyYdc8991T6GAAujiM+ACq1bt06hYSE6OzZs3K5XLr99ts1Z84c9/yuXbsqMDDQPf35559r//79Zc7PKS4u1oEDB+RwOJSfn6/evXu75wUEBKhXr15lPu76wa5du+Tv76/+/ftXue/9+/fr1KlTuuGGGzzGz5w5o6uuukqSlJ2d7dGHJCUlJV102U888YSmTp2qDz74QNu2bdPixYv15JNP6sMPP1TXrl0lSQsXLtQrr7yi3NxcnT59WmfOnFGPHj2q3ev48eM1cuRI7dy5U4MHD9aIESPUp0+fKq0LAP+H4AOgUtddd50WLVqkwMBAtW7dWgEBnruNpk2bekyfOHFCPXv21N/+9rcyy4qMjKxWDz98dHUpTpw4IUn6+9//rjZt2njMCwoKqlYfP9a8eXPdeuutuvXWW/Xkk0/qqquu0jPPPKNly5bpjTfe0PTp0/Xss88qKSlJoaGh+sMf/qBt27ZVu9ehQ4fq0KFDWr9+vTZs2KCBAwcqPT1dzzzzzGU/F8AkBB8AlWratKk6dOhQ5fqf/vSnevPNNxUVFSW73V5uTatWrbRt2zb169dPknTu3Dnt2LFDP/3pT8ut79q1q1wul7KysjRo0KAy83844lRaWuoeS0xMVFBQkHJzcys8UpSQkOA+UfsHn3zyycWfZDmPf8UVV7i/1fXRRx+pT58+mjBhgrvmwIEDFd6/Kr1K54Pj6NGjNXr0aPXt21czZswg+ACXiG91AfCqO+64Qy1atNDw4cO1ZcsWHTx4UJs3b9YDDzygb775RpI0adIkPfXUU3rrrbf05ZdfasKECZVegyc+Pl6jR4/Wr371K7311lvuZS5fvlyS1LZtW9lsNq1bt06FhYU6ceKEQkNDNX36dE2ZMkXLli3TgQMHtHPnTv3pT3/SsmXLJEnjxo3Tvn37NGPGDOXk5Oj111/X0qVLK31+69at05133ql169bpq6++Uk5Ojp555hmtX79ew4cPlyR17NhR27dv13vvvaevvvpKDz/8sD777LMKl1mVXh955BG9/fbb2r9/v/bu3at169YpISGhqv8sAL5H8AHgVU2aNNGHH36ouLg4paSkKCEhQffee6+Ki4vdR4CmTZumu+66S6NHj3Z/FHTLLbdUutxFixYpNTVVEyZMUOfOnfXrX//afYSlTZs2evTRRzVz5ky1bNlSEydOlCT9/ve/18MPP6yMjAwlJCQoOTlZf//7391fO4+Li9OqVav01ltvqXv37u5zdSqTmJioJk2aaNq0aerRo4d+/vOfa/ny5Xr55Zd11113SZJ+85vfKCUlRaNGjVLv3r317bffehz9Kc/Feg0MDNSsWbPUrVs39evXT/7+/nrjjTcu8q8B4EI2q6KzCQEAABoYjvgAAABjEHwAAIAxCD4AAMAYBB8AAGAMgg8AADAGwQcAABiD4AMAAIxB8AEAAMYg+AAAAGMQfAAAgDEIPgAAwBgEHwAAYIz/D3Yko7fysMOtAAAAAElFTkSuQmCC\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import joblib\n",
        "joblib.dump(model, 'linear_regression_model.pkl')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "pZzCoLRkhEGO",
        "outputId": "d121c284-cd4b-4986-89a3-53007b256c65"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "['linear_regression_model.pkl']"
            ]
          },
          "metadata": {},
          "execution_count": 21
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "EyI6tbqvhThN"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}
