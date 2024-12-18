import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import matplotlib.dates as mdates

class StockHMM:
    def __init__(self, n_states):
        self.n_states = n_states
        
        self.A = np.random.rand(n_states, n_states)
        self.A /= self.A.sum(axis=1)[:, np.newaxis]
        
        self.pi = np.random.rand(n_states)
        self.pi /= self.pi.sum()
        
        self.means = np.zeros(n_states)
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
            
            return prices_normalized, df
        
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return None, None

    def _gaussian_emission(self, x, mu, sigma):
        sigma = max(sigma, 1e-6)
        return norm.pdf(x, loc=mu, scale=sigma)

    def baum_welch(self, observations, max_iter=100, tolerance=1e-4):
        T = len(observations)
        
        for iter in range(max_iter):
            alphas = self._forward(observations)
            betas = self._backward(observations)
            
            gamma = self._compute_gamma(alphas, betas)
            xi = self._compute_xi(observations, alphas, betas)
            
            new_A = np.sum(xi, axis=0) / np.sum(np.sum(xi, axis=0), axis=1)[:, np.newaxis]
            
            new_pi = gamma[0]
            
            new_means = np.zeros(self.n_states)
            new_stds = np.zeros(self.n_states)
            
            for k in range(self.n_states):
                weights = gamma[:, k]
                new_means[k] = np.sum(weights * observations) / np.sum(weights)
                new_stds[k] = np.sqrt(np.sum(weights * (observations - new_means[k])**2) / np.sum(weights))
            
            new_stds = np.maximum(new_stds, 1e-6)
            
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

    def _forward(self, observations):
        T = len(observations)
        alphas = np.zeros((T, self.n_states))
        
        for i in range(self.n_states):
            alphas[0, i] = self.pi[i] * self._gaussian_emission(observations[0], self.means[i], self.stds[i])
        
        for t in range(1, T):
            for j in range(self.n_states):
                alphas[t, j] = np.sum(alphas[t-1, :] * self.A[:, j]) * \
                               self._gaussian_emission(observations[t], self.means[j], self.stds[j])
        
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
        
        return betas

    def _compute_gamma(self, alphas, betas):
        gamma = alphas * betas
        gamma /= gamma.sum(axis=1)[:, np.newaxis]
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
            
            xi[t] /= xi[t].sum()
        
        return xi

    def viterbi(self, observations):
        T = len(observations)
        
        log_delta = np.zeros((T, self.n_states))
        log_psi = np.zeros((T, self.n_states), dtype=int)
        
        log_pi = np.log(self.pi + 1e-10)
        log_A = np.log(self.A + 1e-10)
        
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

    def analyze_state_sequence(self, observations, state_sequence):
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
            'transition_matrix': transition_matrix
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
        predictions = predictions[1:]
        
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
        predictions = predictions[1:]
        
        min_length = min(len(predictions), len(original_prices))
        predictions = predictions[:min_length]
        original_prices = original_prices[-min_length:]
        dates = dates[-min_length:]
        
        prediction_dates = pd.date_range(start=dates[-1], periods=len(predictions), freq='D')
        
        metrics = self.calculate_prediction_metrics(original_prices, predictions)
        
        plt.figure(figsize=(16, 12))
        
        plt.subplot(2, 1, 1)
        plt.plot(dates, original_prices, label='Original Prices', color='blue')
        plt.plot(prediction_dates, predictions, label='HMM Predictions', color='red', linestyle='--')
        plt.title('Stock Price: Original vs HMM Predictions', fontsize=15)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Adjusted Close Price', fontsize=12)
        plt.legend(fontsize=10)
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.7)
        
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
        
        plt.subplot(2, 1, 2)
        error = np.abs(original_prices - predictions) / original_prices * 100
        
        plt.plot(dates, error, label='Absolute Percentage Error', color='green')
        plt.title('Prediction Performance Metrics', fontsize=15)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Absolute Percentage Error (%)', fontsize=12)
        
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

def main():
    hmm = StockHMM(n_states=3)
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(project_root, 'data', 'stocks.csv')
    
    normalized_prices, df = hmm.load_stock_data_from_csv(file_path)
    
    if normalized_prices is not None and df is not None:
        hmm.baum_welch(normalized_prices)
        
        state_sequence, log_probability = hmm.viterbi(normalized_prices)
        
        state_analysis = hmm.analyze_state_sequence(normalized_prices, state_sequence)
        
        print("\n--- Viterbi State Sequence Analysis ---")
        print(f"Log Probability of Most Likely Path: {log_probability}")
        
        print("\nState Distribution:")
        for detail in state_analysis['state_details']:
            print(f"State {detail['state']}:")
            print(f"  Market Condition: {detail['market_condition']}")
            print(f"  Mean: {detail['mean']:.4f}")
            print(f"  Volatility: {detail['volatility']:.4f}")
            print(f"  Occurrence: {detail['occurrence_percentage']:.2f}%")
        
        print("\nTransition Matrix:")
        print(state_analysis['transition_matrix'])
        
        last_price_normalized = normalized_prices[-1]
        last_original_price = df['Adj Close'].values[-1]
        
        predictions = hmm.simulate_price_prediction(
            last_price_normalized, 
            n_steps=5, 
            original_prices=df['Adj Close'].values
        )
        
        hmm.visualize_results(
            df['Adj Close'].values, 
            predictions, 
            df['Date'].values
        )

if __name__ == "__main__":
    main()