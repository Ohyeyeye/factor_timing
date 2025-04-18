import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from main import FactorTimingStrategy
from sklearn.metrics import accuracy_score, f1_score

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('strategy_run.log'),
        logging.StreamHandler()
    ]
)

def run_parameter_grid():
    """Run strategy with different parameter combinations"""
    # Define parameter grid
    model_types = ['lstm', 'xgboost', 'hmm']
    optimizer_types = ['neural_regime', 'autoencoder_regime']  # Focus on regime-aware optimizers
    n_regimes_list = [3, 5, 7]  # Test different numbers of regimes
    
    # Store results
    results = []
    
    # Get the correct data directory path
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    
    for model_type in model_types:
        for optimizer_type in optimizer_types:
            for n_regimes in n_regimes_list:
                logging.info(f"\nRunning strategy with {model_type} classifier, {optimizer_type} optimizer, and {n_regimes} regimes")
                
                try:
                    # Initialize strategy
                    logging.info("Initializing strategy...")
                    strategy = FactorTimingStrategy(
                        train_start_date='2014-01-01',
                        train_end_date='2019-12-31',
                        test_start_date='2020-01-01',
                        test_end_date='2024-12-31',
                        model_type=model_type,
                        optimizer_type=optimizer_type,
                        data_dir=data_dir
                    )
                    
                    # Update regime classifier parameters
                    if hasattr(strategy.regime_classifier, 'num_regimes'):
                        strategy.regime_classifier.num_regimes = n_regimes
                    
                    # Update optimizer parameters
                    if hasattr(strategy.portfolio_optimizer, 'n_regimes'):
                        strategy.portfolio_optimizer.n_regimes = n_regimes
                    
                    # Run strategy
                    logging.info("Running strategy...")
                    strategy_results = strategy.run_strategy()
                    
                    # Get regime-specific statistics
                    regime_stats = {}
                    if hasattr(strategy.portfolio_optimizer, 'get_regime_stats'):
                        regime_stats = strategy.portfolio_optimizer.get_regime_stats()
                        logging.info("\nRegime-specific statistics:")
                        for regime, stats in regime_stats.items():
                            logging.info(f"\nRegime {regime}:")
                            logging.info(f"Number of samples: {stats.get('n_samples', 0)}")
                            logging.info(f"Average return: {stats.get('avg_return', 0):.4f}")
                            logging.info(f"Volatility: {stats.get('volatility', 0):.4f}")
                            logging.info(f"Sharpe ratio: {stats.get('sharpe_ratio', 0):.4f}")
                    
                    # Store results
                    results.append({
                        'model_type': model_type,
                        'optimizer_type': optimizer_type,
                        'n_regimes': n_regimes,
                        'results': strategy_results,
                        'regime_stats': regime_stats
                    })
                    
                    # Plot results
                    logging.info("Plotting results...")
                    strategy.plot_results()
                    plt.title(f'{model_type.upper()} Classifier with {optimizer_type.replace("_", " ").title()} Optimizer ({n_regimes} regimes)')
                    plt.savefig(f'results_{model_type}_{optimizer_type}_{n_regimes}.png')
                    plt.close()
                    
                    logging.info(f"Successfully completed run for {model_type} with {optimizer_type} and {n_regimes} regimes")
                    
                except Exception as e:
                    logging.error(f"Error running {model_type} with {optimizer_type} and {n_regimes} regimes: {str(e)}", exc_info=True)
                    continue
    
    if not results:
        logging.error("No valid results obtained from parameter grid search")
        return None
    
    return results

def analyze_results(results: List[Dict]):
    """Analyze and compare results from different parameter combinations"""
    if not results:
        logging.error("No results to analyze")
        return None
        
    # Create DataFrame for analysis
    analysis_data = []
    
    for result in results:
        try:
            metrics = result['results']
            if not isinstance(metrics, dict):
                raise ValueError(f"Invalid metrics format for {result['model_type']} with {result['optimizer_type']}")
                
            # Validate required metrics
            required_metrics = ['sharpe_ratio', 'annualized_return', 'annualized_volatility', 'max_drawdown']
            for metric in required_metrics:
                if metric not in metrics:
                    raise ValueError(f"Missing metric {metric} for {result['model_type']} with {result['optimizer_type']}")
                if not isinstance(metrics[metric], (int, float)):
                    raise ValueError(f"Invalid {metric} value for {result['model_type']} with {result['optimizer_type']}")
            
            # Extract model performance metrics if available
            model_metrics = metrics.get('model_metrics', {})
            accuracy = model_metrics.get('accuracy', None)
            fscore = model_metrics.get('fscore', None)
            
            analysis_data.append({
                'Model': result['model_type'],
                'Optimizer': result['optimizer_type'],
                'Regimes': result['n_regimes'],
                'Sharpe Ratio': metrics['sharpe_ratio'],
                'Annualized Return': metrics['annualized_return'],
                'Annualized Volatility': metrics['annualized_volatility'],
                'Max Drawdown': metrics['max_drawdown'],
                'Model Accuracy': accuracy,
                'F-score': fscore
            })
        except Exception as e:
            logging.error(f"Error processing results for {result.get('model_type', 'unknown')} with {result.get('optimizer_type', 'unknown')}: {str(e)}")
            continue
    
    if not analysis_data:
        logging.error("No valid results to analyze")
        return None
        
    analysis_df = pd.DataFrame(analysis_data)
    
    # Print results
    logging.info("\nStrategy Performance Comparison:")
    logging.info(analysis_df.to_string(index=False))
    
    try:
        # Create figure with subplots
        plt.figure(figsize=(20, 15))
        
        # Financial metrics plots
        plt.subplot(3, 2, 1)
        sns.barplot(data=analysis_df, x='Model', y='Sharpe Ratio', hue='Optimizer')
        plt.title('Sharpe Ratio Comparison')
        plt.xticks(rotation=45)
        
        plt.subplot(3, 2, 2)
        sns.barplot(data=analysis_df, x='Model', y='Annualized Return', hue='Optimizer')
        plt.title('Annualized Return Comparison')
        plt.xticks(rotation=45)
        
        plt.subplot(3, 2, 3)
        sns.barplot(data=analysis_df, x='Model', y='Annualized Volatility', hue='Optimizer')
        plt.title('Volatility Comparison')
        plt.xticks(rotation=45)
        
        plt.subplot(3, 2, 4)
        sns.barplot(data=analysis_df, x='Model', y='Max Drawdown', hue='Optimizer')
        plt.title('Max Drawdown Comparison')
        plt.xticks(rotation=45)
        
        # Model performance metrics plots
        plt.subplot(3, 2, 5)
        sns.barplot(data=analysis_df, x='Model', y='Model Accuracy', hue='Optimizer')
        plt.title('Model Accuracy Comparison')
        plt.xticks(rotation=45)
        
        plt.subplot(3, 2, 6)
        sns.barplot(data=analysis_df, x='Model', y='F-score', hue='Optimizer')
        plt.title('F-score Comparison')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('parameter_comparison.png')
        plt.close()
        
        # Plot regime-specific performance
        for result in results:
            if result.get('regime_stats'):
                plt.figure(figsize=(10, 6))
                regime_performance = []
                for regime, stats in result['regime_stats'].items():
                    if 'model_summary' in stats:
                        regime_performance.append({
                            'Regime': regime,
                            'Parameters': stats['num_parameters']
                        })
                if regime_performance:
                    regime_df = pd.DataFrame(regime_performance)
                    sns.barplot(data=regime_df, x='Regime', y='Parameters')
                    plt.title(f'Model Complexity by Regime ({result["model_type"]} with {result["optimizer_type"]})')
                    plt.savefig(f'regime_complexity_{result["model_type"]}_{result["optimizer_type"]}.png')
                    plt.close()
        
    except Exception as e:
        logging.error(f"Error creating plots: {str(e)}")
    
    return analysis_df

def main():
    try:
        logging.info("Starting factor timing strategy optimization...")
        
        # Run strategy with different parameters
        logging.info("Running parameter grid search...")
        results = run_parameter_grid()
        
        if not results:
            logging.error("No valid results obtained from parameter grid search")
            return
            
        # Analyze and compare results
        logging.info("Analyzing results...")
        analysis_df = analyze_results(results)
        
        if analysis_df is not None:
            # Save results to CSV
            analysis_df.to_csv('strategy_results.csv', index=False)
            logging.info("Results saved to strategy_results.csv")
            
            # Print summary statistics
            logging.info("\nSummary Statistics:")
            for metric in ['Sharpe Ratio', 'Annualized Return', 'Annualized Volatility', 'Max Drawdown']:
                logging.info(f"\n{metric} Statistics:")
                logging.info(f"Mean: {analysis_df[metric].mean():.4f}")
                logging.info(f"Std: {analysis_df[metric].std():.4f}")
                logging.info(f"Min: {analysis_df[metric].min():.4f}")
                logging.info(f"Max: {analysis_df[metric].max():.4f}")
                
            # Find best performing combination
            best_sharpe_idx = analysis_df['Sharpe Ratio'].idxmax()
            best_combo = analysis_df.loc[best_sharpe_idx]
            logging.info("\nBest Performing Combination:")
            logging.info(f"Model: {best_combo['Model']}")
            logging.info(f"Optimizer: {best_combo['Optimizer']}")
            logging.info(f"Regimes: {best_combo['Regimes']}")
            logging.info(f"Sharpe Ratio: {best_combo['Sharpe Ratio']:.4f}")
            logging.info(f"Annualized Return: {best_combo['Annualized Return']:.4f}")
            
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}", exc_info=True)
    finally:
        logging.info("Factor timing strategy optimization completed")

if __name__ == "__main__":
    main() 