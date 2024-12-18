import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import os
import matplotlib.dates as mdates



class StockHMM:
    
    def __init__(self, n_states):
        self.n_states = n_states
        
        # Transition matrix
        self.A = np.random.rand(n_states, n_states)
        self.A /= self.A.sum(axis=1)[:, np.newaxis]
        
        # Initial state probabilities
        self.pi = np.random.rand(n_states)
        self.pi /= self.pi.sum()
        
        # Emission parameters
        self.means = np.random.uniform(low=-1, high=1, size=n_states)  # Random initialization
        self.stds = np.ones(n_states)

    def load_stock_data_from_csv(self, file_path):
        try:
            df = pd.read_csv(file_path)
            required_columns = ['Date', 'Adj Close']
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Required column '{col}' not found in the CSV")
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            prices = df['Adj Close'].values
            prices_normalized = (prices - prices.mean()) / prices.std()
            
            # Print normalized prices
            print("Normalized Prices:", prices_normalized)
            
            return prices_normalized, df
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return None, None
    
    def _gaussian_emission(self, x, mu, sigma):
        sigma = max(sigma, 1e-6)
        return norm.pdf(x, loc=mu, scale=sigma)# calc la fonction de densité de probabilité
                      #(mormilzed_price, mean, std)
        # norm = calcul de probabilités, la génération d'échantillons aléatoires, et plus encore.

    def baum_welch(self, observations, max_iter=100, tolerance=1e-4):
        T = len(observations)
        for iter in range(max_iter):
            alphas = self._forward(observations)
            betas = self._backward(observations)
            gamma = self._compute_gamma(alphas, betas)
            xi = self._compute_xi(observations, alphas, betas)
            
            # Transition matrix update
            new_A = np.sum(xi, axis=0)
            new_A_sum = new_A.sum(axis=1, keepdims=True)
            new_A = new_A / (new_A_sum + 1e-10)  # Avoid division by zero
            
            new_pi = gamma[0]
            new_means = np.zeros(self.n_states)
            new_stds = np.zeros(self.n_states)
            
            for k in range(self.n_states):
                weights = gamma[:, k]
                if np.sum(weights) > 0:  # Check if weights sum to zero
                    new_means[k] = np.sum(weights * observations) / np.sum(weights)
                    new_stds[k] = np.sqrt(np.sum(weights * (observations - new_means[k])**2) / np.sum(weights))
                else:
                    new_means[k] = 0  # Handle case where no observations are assigned to this state
                    new_stds[k] = 1  # Default std deviation or handle as needed
                
            new_stds = np.maximum(new_stds, 1e-6)

            # Print debug information
            print("Iteration:", iter)
            print("Gamma:", gamma)
            print("Xi:", xi)
            print("New A:", new_A)
            print("New Means:", new_means)
            print("New Stds:", new_stds)
            
            if (np.abs(new_A - self.A).max() < tolerance and
                np.abs(new_pi - self.pi).max() < tolerance and
                np.abs(new_means - self.means).max() < tolerance and
                np.abs(new_stds - self.stds).max() < tolerance):
                break
            
            self.A = new_A
            self.pi = new_pi
            self.means = new_means
            self.stds = new_stds
        
        return self.A, self.pi, self.means, self.stds
    
    def fit(self, observations):
        # Check for NaN values in the normalized prices
        if np.any(np.isnan(observations)):
            print("Warning: NaN values found in normalized prices.")
        
        # Print initial parameters
        print("Initial Transition Matrix A:", self.A)
        print("Initial Means:", self.means)
        print("Initial Stds:", self.stds)
        
        self.baum_welch(observations)
    
    def _forward(self, observations):
        T = len(observations)
        alphas = np.zeros((T, self.n_states))
        for i in range(self.n_states):
            alphas[0, i] = self.pi[i] * self._gaussian_emission(observations[0], self.means[i], self.stds[i])
        print("Alpha at t=0:", alphas[0])  # Debugging line
        for t in range(1, T):
            for j in range(self.n_states):
                alphas[t, j] = np.sum(alphas[t-1, :] * self.A[:, j]) * \
                            self._gaussian_emission(observations[t], self.means[j], self.stds[j])
        print("Alphas:", alphas)  # Debugging line
        return alphas

    def _backward(self, observations):
        T = len(observations)
        betas = np.zeros((T, self.n_states))
        betas[T-1, :] = 1
        for t in range(T-2, -1, -1):
            for i in range(self.n_states):
                betas[t, i] = np.sum(self.A[i, :] * 
                                    [self._gaussian_emission(observations[t+1], self.means[j], self.stds[j]) 
                                    * betas[t+1, j] for j in range(self.n_states)])
        print("Betas:", betas)  # Debugging line
        return betas

    def _compute_gamma(self, alphas, betas):
        gamma = alphas * betas
        # Check for zero sums and avoid division by zero
        gamma_sum = gamma.sum(axis=1)[:, np.newaxis]
        gamma_sum[gamma_sum == 0] = 1e-10  # Replace zero sums with a small value
        gamma /= gamma_sum
        return gamma

    def _compute_xi(self, observations, alphas, betas):
        T = len(observations)
        xi = np.zeros((T-1, self.n_states, self.n_states))
        for t in range(T-1):
            for i in range(self.n_states):
                for j in range(self.n_states):
                    xi[t, i, j] = (alphas[t, i] * 
                                self.A[i, j] * 
                                self._gaussian_emission(observations[t+1], self.means[j], self.stds[j]) * 
                                betas[t+1, j])
            xi[t] /= xi[t].sum() if xi[t].sum() != 0 else 1e-10  # Avoid division by zero
        return xi

# trouver la séquence d'états cachés la plus probable dans un HMM
    def viterbi(self, observations):
        T = len(observations) # longueur de la séquence d'observations.
        log_delta = np.zeros((T, self.n_states)) #  matrice qui stocke les probabilités logarithmiques maximales des états à chaque étape de temps.
        log_psi = np.zeros((T, self.n_states), dtype=int) # matrice qui stocke les indices des états précédents qui ont conduit à la probabilité maximale
        
        log_pi = np.log(self.pi + 1e-10)# ogarithme des probabilités initiales des états
        log_A = np.log(self.A + 1e-10)# le logarithme des probabilités de transition entre les états
        
        for i in range(self.n_states):
            log_delta[0, i] = log_pi[i] + np.log(
                self._gaussian_emission(observations[0], self.means[i], self.stds[i]) + 1e-10
            )
        
        for t in range(1, T):
            for j in range(self.n_states):
                transition_probs = log_delta[t-1, :] + log_A[:, j]
                log_delta[t, j] = np.max(transition_probs) + np.log(
                    self._gaussian_emission(observations[t], self.means[j], self.stds[j]) + 1e-10
                )
                log_psi[t, j] = np.argmax(transition_probs)
        
        state_sequence = np.zeros(T, dtype=int)
        state_sequence[T-1] = np.argmax(log_delta[T-1, :])
        
        for t in range(T-2, -1, -1):
            state_sequence[t] = log_psi[t+1, state_sequence[t+1]]
        
        log_probability = np.max(log_delta[T-1, :])
        
        return state_sequence, log_probability

    def analyze_state_sequence(self, observations, state_sequence, log_probability):
        state_counts = np.bincount(state_sequence, minlength=self.n_states)
        
        state_means = []
        state_volatilities = []
        
        for state in range(self.n_states):
            state_observations = observations[state_sequence == state]
            state_means.append(np.mean(state_observations))
            state_volatilities.append(np.std(state_observations))
        
        transition_matrix = np.zeros((self.n_states, self.n_states))
        for t in range(1, len(state_sequence)):
            transition_matrix[state_sequence[t-1], state_sequence[t]] += 1
        
        transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)
        
        state_interpretations = []
        for i, (mean, volatility) in enumerate(zip(state_means, state_volatilities)):
            if volatility < np.median(state_volatilities):
                condition = "Stable"
            elif mean > 0:
                condition = "Bullish"
            else:
                condition = "Bearish"
            
            state_interpretations.append({
                'state': i,
                'mean': mean,
                'volatility': volatility,
                'market_condition': condition,
                'occurrence_percentage': state_counts[i] / len(state_sequence) * 100
            })
        
        state_interpretations.sort(key=lambda x: x['mean'])
        
        return {
            'state_count': dict(zip(range(self.n_states), state_counts)),
            'state_details': state_interpretations,
            'transition_matrix': transition_matrix,
            'log_probability': log_probability  # Include log probability here
        }
   
    def predict_next_state(self, last_observation):
        state_probabilities = np.zeros(self.n_states)
        
        for i in range(self.n_states):
            state_probabilities[i] = np.sum(
                self.A[:, i] * [self._gaussian_emission(last_observation, self.means[j], self.stds[j]) 
                                for j in range(self.n_states)]
            )
        
        return np.argmax(state_probabilities)

    def simulate_price_prediction(self, last_observation, n_steps=5, original_prices=None):
        predictions = [last_observation]
        current_observation = last_observation
        
        for _ in range(n_steps):
            next_state = self.predict_next_state(current_observation)
            
            next_price = np.random.normal(
                loc=self.means[next_state], 
                scale=max(self.stds[next_state], 1e-6)
            )
            
            predictions.append(next_price)
            current_observation = next_price
        
        if original_prices is not None:
            mean = np.mean(original_prices)
            std = np.std(original_prices)
            predictions = [p * std + mean for p in predictions]
        return predictions

    def calculate_prediction_metrics(self, original_prices, predictions):
        # Remove the first prediction as it corresponds to the initial observation
        predictions = predictions[1:]
        
        # Ensure both arrays are of the same length for metric calculation
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
        # Remove the first prediction as it corresponds to the initial observation
        predictions = predictions[1:]
        
        # Ensure all arrays are of the same length
        min_length = min(len(predictions), len(original_prices))
        predictions = predictions[:min_length]
        original_prices = original_prices[-min_length:]
        dates = dates[-min_length:]
        
        # Create prediction dates
        prediction_dates = pd.date_range(start=dates[-1] + pd.Timedelta(days=1), periods=len(predictions), freq='D')
        
        # Calculate metrics for visualization
        metrics = self.calculate_prediction_metrics(original_prices, predictions)
        
        plt.figure(figsize=(16, 12))
        
        # Plot original prices vs predictions
        plt.subplot(2, 1, 1)
        plt.plot(dates, original_prices, label='Original Prices', color='blue')
        plt.plot(prediction_dates, predictions, label='HMM Predictions', color='red', linestyle='--')
        plt.title('Stock Price: Original vs HMM Predictions', fontsize=15)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Adjusted Close Price', fontsize=12)
        plt.legend(fontsize=10)
        
        # Set x-ticks to show every day
        plt.xticks(rotation=45)
        plt.gca().xaxis.set_major_locator(mdates.DayLocator())  # Show every day
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Format the date
        
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Annotate last actual and predicted prices
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
        
        # Plot absolute percentage error
        plt.subplot(2, 1, 2)
        error = np.abs (original_prices - predictions) / original_prices * 100
        plt.plot(dates, error, label='Absolute Percentage Error', color='green')
        plt.title('Prediction Performance Metrics', fontsize=15)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Absolute Percentage Error (%)', fontsize=12)
        
        # Display metrics in the plot
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
        plt.savefig('stock_prediction_results.png', dpi=300)
        plt.close()

def generate_html_report(analysis_results, metrics):
    html_content = f"""
    <html>
    <head>
        <title>Stock HMM Analysis Report</title>
       <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #333; }}
            h2 {{ color: #555; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
            th {{ background-color: #f2f2f2; }}
            img {{ width: 100%; height: auto; display: block; margin: 20px auto; }}  /* Adjusted to 100% width */
            .transition-matrix {{ border: 2px solid #333; border-radius: 5px; }}
            .positive {{ background-color: #c8e6c9; }}
            .negative {{ background-color: #ffcdd2; }}
        </style>
    </head>
    <body>
        <h1>Stock HMM Analysis Report</h1>
        <h2>State Sequence Analysis</h2>
        <p>Log Probability of Most Likely Path: {analysis_results['log_probability']:.4f}</p>
        
        <h3>State Distribution:</h3>
        <table>
            <tr>
                <th>State</th>
                <th>Market Condition</th>
                <th>Mean</th>
                <th>Volatility</th>
                <th>Occurrence (%)</th>
            </tr>
    """
    
    for detail in analysis_results['state_details']:
        html_content += f"""
            <tr>
                <td>{detail['state']}</td>
                <td>{detail['market_condition']}</td>
                <td>{detail['mean']:.4f}</td>
                <td>{detail['volatility']:.4f}</td>
                <td>{detail['occurrence_percentage']:.2f}</td>
            </tr>
        """
    
    html_content += """
        </table>
        <h3>Transition Matrix:</h3>
        <table class="transition-matrix">
            <tr>
                <th>From/To</th>
    """
    
    for i in range (len(analysis_results['transition_matrix'])):
        html_content += f"<th>State {i}</th>"
    
    html_content += "</tr>"
    
    for i, row in enumerate(analysis_results['transition_matrix']):
        html_content += f"<tr><td>State {i}</td>"
        for value in row:
            class_name = "positive" if value > 0.5 else "negative"
            html_content += f"<td class='{class_name}'>{value:.2f}</td>"
        html_content += "</tr>"
    
    html_content += f"""
        </table>
        <h2>Prediction Metrics</h2>
        <p>MAPE: {metrics['MAPE']:.4f}</p>
        <p>MSE: {metrics['MSE']:.4f}</p>
        <p>RMSE: {metrics['RMSE']:.4f}</p>
        <p>Mean Absolute Error: {metrics['Mean Absolute Error']:.4f}</p>
        <p>Relative Error: {metrics['Relative Error']:.4f}</p>
        <h2>Stock Price Predictions</h2>
        <img src="stock_prediction_results.png" alt="Stock Price Predictions">
    </body>
    </html>
    """
    
    return html_content

def main():
    # Load the data
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(project_root, 'data', 'stocks.csv')
    
    n_states = 3  # Define the number of hidden states
    hmm = StockHMM(n_states)
    normalized_prices, df = hmm.load_stock_data_from_csv(file_path)
    
    if normalized_prices is not None:
        hmm.fit(normalized_prices)
        state_sequence, log_probability = hmm.viterbi(normalized_prices)
        
        # Pass log_probability to analyze_state_sequence
        analysis_results = hmm.analyze_state_sequence(normalized_prices, state_sequence, log_probability)

        # Generate predictions
        last_observation = normalized_prices[-1]
        predictions = hmm.simulate_price_prediction(last_observation, n_steps=10, original_prices=normalized_prices)

        # Calculate metrics
        metrics = hmm.calculate_prediction_metrics(normalized_prices, predictions)

        # Generate HTML report
        html_content = generate_html_report(analysis_results, metrics)

        # Save the HTML report
        report_file_path = os.path.join(project_root, 'stock_hmm_report.html')
        with open(report_file_path, 'w') as f:
            f.write(html_content)

        print(f"Report saved to {report_file_path}")

        # Visualize results
        hmm.visualize_results(normalized_prices, predictions, df['Date'].values)

if __name__ == "__main__":
    main()
