# Flight Price Prediction

This project aims to analyze a flight booking dataset obtained from the “Ease My Trip” website to derive meaningful insights and predict flight prices using various statistical and machine learning techniques.

## Dataset

The dataset contains information about flight booking options for travel between India's top 6 metro cities. It includes 300,261 data points and 11 features. The features are as follows:

1. **Airline**: The name of the airline company.
2. **Flight**: Flight code.
3. **Source City**: City from which the flight takes off.
4. **Departure Time**: Time of departure, grouped into bins.
5. **Stops**: Number of stops between the source and destination cities.
6. **Arrival Time**: Time of arrival, grouped into bins.
7. **Destination City**: City where the flight will land.
8. **Class**: Seat class, either Business or Economy.
9. **Duration**: Total travel time between cities in hours.
10. **Days Left**: Days left for the journey from the booking date.
11. **Price**: Target variable, the price of the ticket.

## Project Structure

- `flight_price_prediction.ipynb`: Jupyter Notebook containing the data exploration, preprocessing, EDA, and model building steps.
- `Clean_Dataset.csv`: The dataset used for analysis and model building.
- Download data from the `https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction/data`
- `README.md`: This readme file.

## Getting Started

To get started with this project, follow these steps:

### Prerequisites

- Python 3.7 or above
- Jupyter Notebook or Jupyter Lab
- The following Python packages:
  - pandas
  - numpy
  - seaborn
  - matplotlib
  - plotly
  - scikit-learn
  - statsmodels

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/your-username/flight-price-prediction.git
    cd flight-price-prediction
    ```

2. Install the required packages:
    ```sh
    pip install pandas numpy seaborn matplotlib plotly scikit-learn statsmodels
    ```

### Usage

1. Open the Jupyter Notebook:
    ```sh
    jupyter notebook flight_price_prediction.ipynb
    ```

2. Follow the steps in the notebook to:
   - Load and explore the dataset.
   - Preprocess the data.
   - Perform exploratory data analysis (EDA).
   - Build and evaluate regression models to predict flight prices.
   - Check assumptions and improve the models.

## Exploratory Data Analysis (EDA)

In this project, we performed EDA to understand the distribution of data and relationships between features. Some of the visualizations include:
- Distribution of airlines and their average pricing.
- Class distribution in airlines.
- Price trends based on departure time and arrival time.
- Analysis of business and economy class flights.

## Modeling

We used Linear Regression and Statsmodels to build and evaluate models for flight price prediction. We also checked assumptions like normality and equal variance to ensure model reliability. Additionally, we explored transforming the target variable to improve model performance.

## Results

The models were evaluated using metrics like R-squared and Root Mean Square Error (RMSE). The final model was selected based on the assumption checks and performance metrics.

## Contributing

If you have suggestions or improvements, feel free to create a pull request or open an issue.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

