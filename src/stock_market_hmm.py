import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import matplotlib.dates as mdates

class StockHMM:
    def __init__(self, n_states):
        """
        Initialize Hidden Markov Model for stock price prediction
        
        Parameters:
        - n_states: Number of hidden states to model market conditions
        """
        self.n_states = n_states
        
        # Transition probabilities between hidden states
        self.A = np.random.rand(n_states, n_states)
        self.A /= self.A.sum(axis=1)[:, np.newaxis]
        
        # Initial state probabilities
        self.pi = np.random.rand(n_states)
        self.pi /= self.pi.sum()
        
        # Emission parameters (mean and standard deviation for each state)
        self.means = np.zeros(n_states)
        self.stds = np.ones(n_states)  # Initialize with 1 to avoid zero division

    def load_stock_data_from_csv(self, file_path):
        """
        Load stock price data from a CSV file
        
        Parameters:
        - file_path: Path to the CSV file
        
        Returns:
        Normalized prices, original dataframe
        """
        try:
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Check required columns
            required_columns = ['Date', 'Adj Close']
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Required column '{col}' not found in the CSV")
            
            # Convert Date column to datetime
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Sort by date
            df = df.sort_values('Date')
            
            # Extract adjusted closing prices
            prices = df['Adj Close'].values
            
            # Normalize prices
            prices_normalized = (prices - prices.mean()) / prices.std()
            
            return prices_normalized, df
        
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return None, None

    def _gaussian_emission(self, x, mu, sigma):
        """
        Gaussian probability density function for emission
        """
        # Add small epsilon to prevent division by zero
        sigma = max(sigma, 1e-6)
        return norm.pdf(x, loc=mu, scale=sigma)

    def baum_welch(self, observations, max_iter=100, tolerance=1e-4):
        """
        Baum-Welch algorithm for HMM parameter estimation
        """
        T = len(observations)
        
        # Initialization
        for iter in range(max_iter):
            # Expectation step: Forward-Backward algorithm
            alphas = self._forward(observations)
            betas = self._backward(observations)
            
            # Compute gamma and xi
            gamma = self._compute_gamma(alphas, betas)
            xi = self._compute_xi(observations, alphas, betas)
            
            # Maximization step: Update parameters
            # Update transition probabilities
            new_A = np.sum(xi, axis=0) / np.sum(np.sum(xi, axis=0), axis=1)[:, np.newaxis]
            
            # Update initial state probabilities
            new_pi = gamma[0]
            
            # Update emission parameters (means and std devs)
            new_means = np.zeros(self.n_states)
            new_stds = np.zeros(self.n_states)
            
            for k in range(self.n_states):
                weights = gamma[:, k]
                new_means[k] = np.sum(weights * observations) / np.sum(weights)
                new_stds[k] = np.sqrt(np.sum(weights * (observations - new_means[k])**2) / np.sum(weights))
            
            # Ensure standard deviations are not zero
            new_stds = np.maximum(new_stds, 1e-6)
            
            # Check convergence
            if (np.abs(new_A - self.A).max() < tolerance and
                np.abs(new_pi - self.pi).max() < tolerance and
                np.abs(new_means - self.means).max() < tolerance and
                np.abs(new_stds - self.stds).max() < tolerance):
                break
            
            # Update model parameters
            self.A = new_A
            self.pi = new_pi
            self.means = new_means
            self.stds = new_stds
        
        return self.A, self.pi, self.means, self.stds

    def _forward(self, observations):
        """
        Forward algorithm (alpha pass)
        """
        T = len(observations)
        alphas = np.zeros((T, self.n_states))
        
        # Initialize first column
        for i in range(self.n_states):
            alphas[0, i] = self.pi[i] * self._gaussian_emission(observations[0], self.means[i], self.stds[i])
        
        # Recursive computation
        for t in range(1, T):
            for j in range(self.n_states):
                alphas[t, j] = np.sum(alphas[t-1, :] * self.A[:, j]) * \
                               self._gaussian_emission(observations[t], self.means[j], self.stds[j])
        
        return alphas

    def _backward(self, observations):
        """
        Backward algorithm (beta pass)
        """
        T = len(observations)
        betas = np.zeros((T, self.n_states))
        
        # Initialize last column
        betas[T-1, :] = 1
        
        # Recursive computation
        for t in range(T-2, -1, -1):
            for i in range(self.n_states):
                betas[t, i] = np.sum(self.A[i, :] * 
                                     [self._gaussian_emission(observations[t+1], self.means[j], self.stds[j]) 
                                      * betas[t+1, j] for j in range(self.n_states)])
        
        return betas

    def _compute_gamma(self, alphas, betas):
        """
        Compute state posterior probabilities
        """
        gamma = alphas * betas
        gamma /= gamma.sum(axis=1)[:, np.newaxis]
        return gamma

    def _compute_xi(self, observations, alphas, betas):
        """
        Compute state transition posterior probabilities
        """
        T = len(observations)
        xi = np.zeros((T-1, self.n_states, self.n_states))
        
        for t in range(T-1):
            for i in range(self.n_states):
                for j in range(self.n_states):
                    xi[t, i, j] = (alphas[t, i] * 
                                   self.A[i, j] * 
                                   self._gaussian_emission(observations[t+1], self.means[j], self.stds[j]) * 
                                   betas[t+1, j])
            
            xi[t] /= xi[t].sum()
        
        return xi

    def predict_next_state(self, last_observation):
        """
        Predict the most likely next state
        """
        state_probabilities = np.zeros(self.n_states)
        
        for i in range(self.n_states):
            # Compute probability of transitioning to each state
            state_probabilities[i] = np.sum(
                self.A[:, i] * [self._gaussian_emission(last_observation, self.means[j], self.stds[j]) 
                                for j in range(self.n_states)]
            )
        
        return np.argmax(state_probabilities)

    def simulate_price_prediction(self, last_observation, n_steps=5, original_prices=None):
        """
        Simulate future price predictions
        """
        predictions = [last_observation]
        current_observation = last_observation
        
        for _ in range(n_steps):
            # Predict next state
            next_state = self.predict_next_state(current_observation)
            
            # Generate next price based on the state's distribution
            next_price = np.random.normal(
                loc=self.means[next_state], 
                scale=max(self.stds[next_state], 1e-6)  # Ensure non-zero scale
            )
            
            predictions.append(next_price)
            current_observation = next_price
        
        # Denormalize predictions if original prices are provided
        if original_prices is not None:
            mean = np.mean(original_prices)
            std = np.std(original_prices)
            predictions = [p * std + mean for p in predictions]
        
        return predictions

    def calculate_prediction_metrics(self, original_prices, predictions):
        """
        Calculate advanced prediction performance metrics.
        
        Parameters:
        - original_prices: Actual stock prices
        - predictions: HMM predicted prices
        
        Returns:
        Dictionary of performance metrics
        """
        # Ensure predictions start from the last price point
        predictions = predictions[1:]
        
        # Truncate to match the minimum length between original prices and predictions
        min_length = min(len(predictions), len(original_prices))
        predictions = predictions[:min_length]
        original_prices = original_prices[-min_length:]
        
        metrics = {
            'MAPE': mean_absolute_percentage_error(original_prices, predictions) * 100,
            'MSE': mean_squared_error(original_prices, predictions),
            'RMSE': np.sqrt(mean_squared_error(original_prices, predictions)),
            'Mean Absolute Error': np.mean(np.abs(original_prices - predictions)),
            'Relative Error': np.abs(original_prices[-1] - predictions[-1]) / original_prices[-1] * 100
        }
        
        return metrics

    def visualize_results(self, original_prices, predictions, dates):
        """
        Enhanced visualization with more detailed insights and statistical annotations.
        
        Parameters:
        - original_prices: Original stock prices
        - predictions: HMM predicted prices
        - dates: Corresponding dates
        """
        # Adjust predictions to match the length of original prices
        predictions = predictions[1:]
        
        # Truncate to match the minimum length between original prices and predictions
        min_length = min(len(predictions), len(original_prices))
        predictions = predictions[:min_length]
        original_prices = original_prices[-min_length:]
        dates = dates[-min_length:]
        
        # Use the last prediction date to generate prediction dates
        prediction_dates = pd.date_range(start=dates[-1], periods=len(predictions), freq='D')
        
        # Recalculate prediction metrics
        metrics = self.calculate_prediction_metrics(original_prices, predictions)
        
        plt.figure(figsize=(16, 12))
        
        # Price Comparison Plot
        plt.subplot(2, 1, 1)
        plt.plot(dates, original_prices, label='Original Prices', color='blue')
        plt.plot(prediction_dates, predictions, label='HMM Predictions', color='red', linestyle='--')
        plt.title('Stock Price: Original vs HMM Predictions', fontsize=15)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Adjusted Close Price', fontsize=12)
        plt.legend(fontsize=10)
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Annotate prediction details
        last_price = original_prices[-1]
        last_prediction = predictions[-1]
        plt.annotate(f'Last Actual Price: ${last_price:.2f}\n'
                     f'Predicted Price: ${last_prediction:.2f}\n'
                     f'Relative Error: {metrics["Relative Error"]:.2f}%', 
                     xy=(dates[-1], last_price), 
                     xytext=(10, -30), 
                     textcoords='offset points', 
                     ha='left', va='top',
                     bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                     fontsize=9)
        
        # Error Plot
        plt.subplot(2, 1, 2)
        error = np.abs(original_prices - predictions) / original_prices * 100
        
        plt.plot(dates, error, label='Absolute Percentage Error', color='green')
        plt.title('Prediction Performance Metrics', fontsize=15)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Absolute Percentage Error (%)', fontsize=12)
        
        # Annotate performance metrics
        metrics_text = (
            f'Mean Absolute % Error (MAPE): {metrics["MAPE"]:.2f}%\n'
            f'Root Mean Square Error (RMSE): ${metrics["RMSE"]:.2f}\n'
            f'Mean Absolute Error: ${metrics["Mean Absolute Error"]:.2f}'
        )
        plt.annotate(metrics_text, 
                     xy=(0.05, 0.95), 
                     xycoords='axes fraction',
                     ha='left', va='top',
                     bbox=dict(boxstyle='round', fc='lightgreen', alpha=0.5),
                     fontsize=9)
        
        plt.legend(fontsize=10)
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('stock_prediction_results.png', dpi=300)  # Higher resolution
        plt.close()

def main():
    # Initialize HMM with 3 hidden states
    hmm = StockHMM(n_states=3)
    
    # Dynamically find the CSV file
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(project_root, 'data', 'stocks.csv')
    
    # Load stock price data from CSV
    normalized_prices, df = hmm.load_stock_data_from_csv(file_path)
    
    if normalized_prices is not None and df is not None:
        # Train the model using Baum-Welch algorithm
        hmm.baum_welch(normalized_prices)
        
        # Get the last observation for prediction
        last_price = normalized_prices[-1]
        
        # Simulate price predictions
        price_predictions = hmm.simulate_price_prediction(
            last_price, 
            n_steps=10, 
            original_prices=df['Adj Close'].values
        )
        
        # Print predictions
        print("Last Observed Price:", df['Adj Close'].values[-1])
        print("Price Predictions:", price_predictions)
        
        # Visualize results
        hmm.visualize_results(
            df['Adj Close'].values, 
            price_predictions, 
            df['Date'].values
        )
        
        print("Visualization saved as stock_prediction_results.png")
    else:
        print("Failed to load stock data")

if __name__ == "__main__":
    main()