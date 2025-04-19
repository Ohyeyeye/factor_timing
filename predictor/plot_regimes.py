import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from data.data_loader import DataLoader

def plot_market_regimes():
    # Initialize data loader
    data_loader = DataLoader()
    
    # Load all data with extended date range
    market_data = data_loader.load_market_data('2020-01-01', '2024-12-31')
    macro_data = data_loader.load_macro_data('2020-01-01', '2024-12-31')
    
    # Resample macro data to daily frequency using forward fill
    macro_data = macro_data.resample('D').ffill()
    
    # Align the data on the intersection of dates
    common_dates = market_data.index.intersection(macro_data.index)
    market_data = market_data.loc[common_dates]
    macro_data = macro_data.loc[common_dates]
    
    # Combine features for clustering
    features = pd.concat([
        market_data,
        macro_data
    ], axis=1)
    
    # Calculate additional features
    returns = market_data['SP500_Return']
    features['momentum'] = returns.rolling(window=60).mean()
    features['volatility'] = returns.rolling(window=20).std() * np.sqrt(252)
    
    # Handle missing values
    features = features.ffill(limit=5).bfill(limit=5).fillna(features.mean())
    features = features.replace([np.inf, -np.inf], [1e10, -1e10])
    
    # Drop any remaining NaN values
    features = features.dropna()
    
    # Scale features for clustering
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    features_scaled = pd.DataFrame(features_scaled, index=features.index, columns=features.columns)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=5, random_state=42)
    regime_labels = kmeans.fit_predict(features_scaled)
    
    # Set style
    plt.style.use('bmh')
    colors = ['#2ecc71', '#e74c3c', '#f1c40f', '#3498db', '#9b59b6']
    
    # Create figure with subplots
    fig, axes = plt.subplots(5, 1, figsize=(20, 30))
    fig.suptitle('Market Regimes Analysis (2020-2024)', fontsize=16, y=0.91)
    
    # Calculate S&P 500 cumulative returns
    sp500_cumulative = (1 + market_data['SP500_Return']).cumprod()
    
    # Create the plots
    plots = [
        (sp500_cumulative, 'S&P 500 (Cumulative Returns)'),
        (market_data['VIX'], 'VIX (Volatility Index)'),
        (macro_data['GDP'], 'GDP (Gross Domestic Product)'),
        (macro_data['CPI'], 'CPI (Consumer Price Index)'),
        (macro_data['UNRATE'], 'Unemployment Rate')
    ]
    
    # Plot each indicator
    for i, (data, title) in enumerate(plots):
        # Plot the main line
        axes[i].plot(data.index, data.values, 'k-', linewidth=1.5, zorder=2)
        
        # Color the background for each regime
        ymin, ymax = data.min(), data.max()
        margin = (ymax - ymin) * 0.1
        ymin -= margin
        ymax += margin
        axes[i].set_ylim(ymin, ymax)
        
        # Find regime change points
        regime_changes = np.where(np.diff(regime_labels) != 0)[0] + 1
        regime_changes = np.concatenate(([0], regime_changes, [len(regime_labels)]))
        
        # Color each regime period
        for j in range(len(regime_changes) - 1):
            start_idx = regime_changes[j]
            end_idx = regime_changes[j + 1]
            regime = regime_labels[start_idx]
            
            axes[i].axvspan(data.index[start_idx], data.index[end_idx-1], 
                          alpha=0.2, color=colors[regime], zorder=1)
        
        axes[i].set_title(title, pad=20, fontsize=12)
        axes[i].grid(True, alpha=0.3, zorder=0)
        
        # Create custom legend for regimes
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=colors[i], alpha=0.2, label=f'Regime {i}')
                         for i in range(5)]
        axes[i].legend(handles=legend_elements, bbox_to_anchor=(1.01, 1), loc='upper left')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('market_regimes.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Print regime statistics
    print("\nRegime Statistics (2020-2024):")
    print("-----------------")
    for regime in range(5):
        mask = regime_labels == regime
        stats = {
            'Count': sum(mask),
            'Avg Return': market_data['SP500_Return'][mask].mean() * 100,
            'Volatility': market_data['VIX'][mask].mean(),
            'GDP': macro_data['GDP'][mask].mean(),
            'CPI': macro_data['CPI'][mask].mean(),
            'Unemployment': macro_data['UNRATE'][mask].mean(),
            'First Date': features.index[mask][0].strftime('%Y-%m-%d'),
            'Last Date': features.index[mask][-1].strftime('%Y-%m-%d')
        }
        print(f"\nRegime {regime}:")
        print(f"Time Range: {stats['First Date']} to {stats['Last Date']}")
        print(f"Occurrences: {stats['Count']} days")
        print(f"Average Return: {stats['Avg Return']:.2f}%")
        print(f"Average VIX: {stats['Volatility']:.2f}")
        print(f"Average GDP: {stats['GDP']:.2f}")
        print(f"Average CPI: {stats['CPI']:.2f}")
        print(f"Average Unemployment: {stats['Unemployment']:.2f}%")

if __name__ == "__main__":
    plot_market_regimes() 