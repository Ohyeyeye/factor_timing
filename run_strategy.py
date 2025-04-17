import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
from main import FactorTimingStrategy

def run_parameter_grid():
    """Run strategy with different parameter combinations"""
    # Define parameter grid
    model_types = ['lstm', 'xgboost', 'hmm']
    optimizer_types = ['mean_variance', 'neural']
    
    # Store results
    results = []
    
    for model_type in model_types:
        for optimizer_type in optimizer_types:
            print(f"\nRunning strategy with {model_type} classifier and {optimizer_type} optimizer")
            
            try:
                # Initialize strategy
                strategy = FactorTimingStrategy(
                    train_start_date='2014-01-01',
                    train_end_date='2019-12-31',
                    test_start_date='2020-01-01',
                    test_end_date='2024-12-31',
                    model_type=model_type,
                    optimizer_type=optimizer_type
                )
                
                # Run strategy
                strategy_results = strategy.run_strategy()
                
                # Store results
                results.append({
                    'model_type': model_type,
                    'optimizer_type': optimizer_type,
                    'results': strategy_results
                })
                
                # Plot results
                strategy.plot_results()
                plt.title(f'{model_type.upper()} Classifier with {optimizer_type.replace("_", " ").title()} Optimizer')
                plt.savefig(f'results_{model_type}_{optimizer_type}.png')
                plt.close()
                
            except Exception as e:
                print(f"Error running {model_type} with {optimizer_type}: {str(e)}")
                continue
    
    return results

def analyze_results(results: List[Dict]):
    """Analyze and compare results from different parameter combinations"""
    # Create DataFrame for analysis
    analysis_data = []
    
    for result in results:
        metrics = result['results']
        analysis_data.append({
            'Model': result['model_type'],
            'Optimizer': result['optimizer_type'],
            'Sharpe Ratio': metrics['sharpe_ratio'],
            'Annualized Return': metrics['annualized_return'],
            'Annualized Volatility': metrics['annualized_volatility'],
            'Max Drawdown': metrics['max_drawdown']
        })
    
    analysis_df = pd.DataFrame(analysis_data)
    
    # Print results
    print("\nStrategy Performance Comparison:")
    print(analysis_df.to_string(index=False))
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    
    # Sharpe Ratio comparison
    plt.subplot(2, 2, 1)
    sns.barplot(data=analysis_df, x='Model', y='Sharpe Ratio', hue='Optimizer')
    plt.title('Sharpe Ratio Comparison')
    
    # Annualized Return comparison
    plt.subplot(2, 2, 2)
    sns.barplot(data=analysis_df, x='Model', y='Annualized Return', hue='Optimizer')
    plt.title('Annualized Return Comparison')
    
    # Volatility comparison
    plt.subplot(2, 2, 3)
    sns.barplot(data=analysis_df, x='Model', y='Annualized Volatility', hue='Optimizer')
    plt.title('Volatility Comparison')
    
    # Max Drawdown comparison
    plt.subplot(2, 2, 4)
    sns.barplot(data=analysis_df, x='Model', y='Max Drawdown', hue='Optimizer')
    plt.title('Max Drawdown Comparison')
    
    plt.tight_layout()
    plt.savefig('parameter_comparison.png')
    plt.close()
    
    return analysis_df

def main():
    # Run strategy with different parameters
    results = run_parameter_grid()
    
    # Analyze and compare results
    analysis_df = analyze_results(results)
    
    # Save results to CSV
    analysis_df.to_csv('strategy_results.csv', index=False)
    print("\nResults saved to strategy_results.csv")

if __name__ == "__main__":
    main() 