from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
from typing import List, Optional, Tuple
import logging

# Zaman serisi analizi icin ek kutuphaneler
try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller, kpss
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

logger = logging.getLogger(__name__)

def exploratory_data_analysis(
    df: pd.DataFrame, 
    output_dir: str = "outputs/png",
    target_col: Optional[str] = None,
    rolling_window: int = 30,
    figsize: Tuple[int, int] = (12, 8)
) -> dict:
    """
    Kapsamli kesifsel veri analizi ve gorsellestirme
    """
    logger.info("EDA analizi basliyor...")
    
    # Cikti dizinini olustur
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    eda_results = {
        "timestamp": datetime.now().isoformat(),
        "generated_plots": [],
        "statistical_tests": {},
        "feature_analysis": {},
        "time_series_analysis": {}
    }
    
    # 1. Temel dagilim grafikleri
    logger.info("Dagilim grafikleri olusturuluyor...")
    _create_distribution_plots(df, output_path, figsize)
    eda_results["generated_plots"].extend([
        "distribution_numeric.png",
        "distribution_categorical.png" if df.select_dtypes(include=['object']).shape[1] > 0 else None
    ])
    
    # 2. Korelasyon matrisi
    logger.info("Korelasyon matrisi olusturuluyor...")
    _create_correlation_matrix(df, output_path, figsize)
    eda_results["generated_plots"].append("correlation_matrix.png")
    
    # 3. Zaman serisi analizi (eger datetime index varsa)
    if pd.api.types.is_datetime64_any_dtype(df.index):
        logger.info("Zaman serisi analizi yapiliyor...")
        ts_results = _time_series_analysis(df, output_path, rolling_window, figsize, target_col)
        eda_results["time_series_analysis"] = ts_results
        eda_results["generated_plots"].extend([
            "time_series_overview.png",
            "rolling_statistics.png",
            "seasonal_decomposition.png" if STATSMODELS_AVAILABLE else None,
            "autocorrelation.png" if STATSMODELS_AVAILABLE else None
        ])
    
    # 4. Feature importance analizi (eger target var ise)
    if target_col and target_col in df.columns:
        logger.info("Feature importance analizi yapiliyor...")
        feature_results = _feature_importance_analysis(df, target_col, output_path, figsize)
        eda_results["feature_analysis"] = feature_results
        eda_results["generated_plots"].append("feature_importance.png")
    
    # 5. Outlier analizi
    logger.info("Outlier analizi yapiliyor...")
    _create_outlier_plots(df, output_path, figsize)
    eda_results["generated_plots"].append("outlier_analysis.png")
    
    # None degerlerini temizle
    eda_results["generated_plots"] = [p for p in eda_results["generated_plots"] if p is not None]
    
    logger.info(f"EDA tamamlandi. {len(eda_results['generated_plots'])} grafik olusturuldu.")
    return eda_results


def _create_distribution_plots(df: pd.DataFrame, output_path: Path, figsize: Tuple[int, int]):
    """Numerik ve kategorik degiskenler icin dagilim grafikleri"""
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # Numerik degiskenler icin histogramlar
    if len(numeric_cols) > 0:
        n_cols = min(4, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], figsize[1] * n_rows / 2))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(numeric_cols):
            if df[col].notna().sum() > 0:
                axes[i].hist(df[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
                axes[i].set_title(f'{col} Dagilimi')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frekans')
                
                # Istatistikleri ekle
                mean_val = df[col].mean()
                std_val = df[col].std()
                axes[i].axvline(mean_val, color='red', linestyle='--', label=f'Ortalama: {mean_val:.2f}')
                axes[i].axvline(mean_val + std_val, color='orange', linestyle='--', alpha=0.7)
                axes[i].axvline(mean_val - std_val, color='orange', linestyle='--', alpha=0.7)
                axes[i].legend()
        
        # Bos subplotlari kapat
        for i in range(len(numeric_cols), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path / "distribution_numeric.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Kategorik degiskenler icin bar grafikleri
    if len(categorical_cols) > 0:
        n_cols = min(3, len(categorical_cols))
        n_rows = (len(categorical_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], figsize[1] * n_rows / 2))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(categorical_cols):
            value_counts = df[col].value_counts().head(10)  # En fazla 10 kategori
            axes[i].bar(range(len(value_counts)), value_counts.values)
            axes[i].set_title(f'{col} Dagilimi')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frekans')
            axes[i].set_xticks(range(len(value_counts)))
            axes[i].set_xticklabels(value_counts.index, rotation=45, ha='right')
        
        # Bos subplotlari kapat
        for i in range(len(categorical_cols), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path / "distribution_categorical.png", dpi=300, bbox_inches='tight')
        plt.close()


def _create_correlation_matrix(df: pd.DataFrame, output_path: Path, figsize: Tuple[int, int]):
    """Korelasyon matrisi isi haritasi"""
    
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.shape[1] > 1:
        plt.figure(figsize=figsize)
        correlation_matrix = numeric_df.corr()
        
        # Maske olustur (ust ucgen)
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        # Isi haritasi olustur
        sns.heatmap(
            correlation_matrix, 
            mask=mask,
            annot=True, 
            cmap='RdBu_r', 
            center=0,
            square=True,
            fmt='.2f',
            cbar_kws={'shrink': 0.8}
        )
        
        plt.title('Korelasyon Matrisi', fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig(output_path / "correlation_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()


def _time_series_analysis(
    df: pd.DataFrame, 
    output_path: Path, 
    rolling_window: int, 
    figsize: Tuple[int, int],
    target_col: Optional[str] = None
) -> dict:
    """Zaman serisi analizi ve gorsellestirme"""
    
    ts_results = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Ana zaman serisi grafigi
    fig, axes = plt.subplots(len(numeric_cols), 1, figsize=(figsize[0], figsize[1] * len(numeric_cols) / 2))
    if len(numeric_cols) == 1:
        axes = [axes]
    
    for i, col in enumerate(numeric_cols):
        axes[i].plot(df.index, df[col], alpha=0.7, label=col)
        axes[i].set_title(f'{col} Zaman Serisi')
        axes[i].set_ylabel(col)
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Tarih')
    plt.tight_layout()
    plt.savefig(output_path / "time_series_overview.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Rolling istatistikler
    fig, axes = plt.subplots(len(numeric_cols), 1, figsize=(figsize[0], figsize[1] * len(numeric_cols) / 2))
    if len(numeric_cols) == 1:
        axes = [axes]
    
    for i, col in enumerate(numeric_cols):
        if df[col].notna().sum() > rolling_window:
            rolling_mean = df[col].rolling(window=rolling_window).mean()
            rolling_std = df[col].rolling(window=rolling_window).std()
            
            axes[i].plot(df.index, df[col], alpha=0.3, label='Orijinal')
            axes[i].plot(df.index, rolling_mean, color='red', label=f'{rolling_window}d Hareketli Ortalama')
            axes[i].fill_between(
                df.index,
                rolling_mean - rolling_std,
                rolling_mean + rolling_std,
                alpha=0.2, color='red', label=f'{rolling_window}d Std Band'
            )
            
            axes[i].set_title(f'{col} - Rolling Istatistikler')
            axes[i].set_ylabel(col)
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Tarih')
    plt.tight_layout()
    plt.savefig(output_path / "rolling_statistics.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Statsmodels analizi (eger mevcut ise)
    if STATSMODELS_AVAILABLE and target_col and target_col in numeric_cols:
        ts_results["stationarity_tests"] = _stationarity_tests(df[target_col].dropna())
        _seasonal_decomposition_plot(df[target_col].dropna(), output_path, figsize)
        _autocorrelation_plots(df[target_col].dropna(), output_path, figsize)
    
    return ts_results


def _stationarity_tests(series: pd.Series) -> dict:
    """Duraganlik testleri"""
    results = {}
    
    try:
        # Augmented Dickey-Fuller Test
        adf_result = adfuller(series)
        results["adf_test"] = {
            "statistic": float(adf_result[0]),
            "p_value": float(adf_result[1]),
            "critical_values": {k: float(v) for k, v in adf_result[4].items()},
            "is_stationary": adf_result[1] < 0.05
        }
        
        # KPSS Test
        kpss_result = kpss(series, regression='c')
        results["kpss_test"] = {
            "statistic": float(kpss_result[0]),
            "p_value": float(kpss_result[1]),
            "critical_values": {k: float(v) for k, v in kpss_result[3].items()},
            "is_stationary": kpss_result[1] > 0.05
        }
        
    except Exception as e:
        logger.warning(f"Duraganlik testleri basarisiz: {str(e)}")
        results["error"] = str(e)
    
    return results


def _seasonal_decomposition_plot(series: pd.Series, output_path: Path, figsize: Tuple[int, int]):
    """Mevsimsel ayristirma grafigi"""
    try:
        # Minimum 2 periyot gerekli
        if len(series) < 24:
            return
        
        decomposition = seasonal_decompose(series, model='additive', period=min(12, len(series)//2))
        
        fig, axes = plt.subplots(4, 1, figsize=(figsize[0], figsize[1] * 1.5))
        
        decomposition.observed.plot(ax=axes[0], title='Orijinal Seri')
        decomposition.trend.plot(ax=axes[1], title='Trend')
        decomposition.seasonal.plot(ax=axes[2], title='Mevsimsellik')
        decomposition.resid.plot(ax=axes[3], title='Artik (Residual)')
        
        for ax in axes:
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / "seasonal_decomposition.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        logger.warning(f"Mevsimsel ayristirma basarisiz: {str(e)}")


def _autocorrelation_plots(series: pd.Series, output_path: Path, figsize: Tuple[int, int]):
    """Otokorelasyon ve kismi otokorelasyon grafikleri"""
    try:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        plot_acf(series, ax=axes[0], lags=min(40, len(series)//4))
        axes[0].set_title('Otokorelasyon Fonksiyonu (ACF)')
        
        plot_pacf(series, ax=axes[1], lags=min(40, len(series)//4))
        axes[1].set_title('Kismi Otokorelasyon Fonksiyonu (PACF)')
        
        plt.tight_layout()
        plt.savefig(output_path / "autocorrelation.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        logger.warning(f"Otokorelasyon grafikleri basarisiz: {str(e)}")


def _feature_importance_analysis(
    df: pd.DataFrame, 
    target_col: str, 
    output_path: Path, 
    figsize: Tuple[int, int]
) -> dict:
    """Feature importance analizi"""
    
    numeric_df = df.select_dtypes(include=[np.number])
    features = [col for col in numeric_df.columns if col != target_col]
    
    if not features or target_col not in numeric_df.columns:
        return {}
    
    # Korelasyon tabanli feature importance
    correlations = numeric_df[features].corrwith(numeric_df[target_col]).abs().sort_values(ascending=False)
    
    plt.figure(figsize=figsize)
    correlations.plot(kind='barh')
    plt.title(f'{target_col} ile Feature Korelasyonlari')
    plt.xlabel('Mutlak Korelasyon')
    plt.tight_layout()
    plt.savefig(output_path / "feature_importance.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        "correlation_based": correlations.to_dict(),
        "top_features": correlations.head(5).index.tolist()
    }


def _create_outlier_plots(df: pd.DataFrame, output_path: Path, figsize: Tuple[int, int]):
    """Outlier analizi icin box plot ve scatter plot"""
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        return
    
    # Box plots
    n_cols = min(4, len(numeric_cols))
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], figsize[1] * n_rows / 2))
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, col in enumerate(numeric_cols):
        if df[col].notna().sum() > 0:
            axes[i].boxplot(df[col].dropna())
            axes[i].set_title(f'{col} Box Plot')
            axes[i].set_ylabel(col)
            
            # Outlier sayisini ekle
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            axes[i].text(0.02, 0.98, f'Outliers: {len(outliers)}', 
                        transform=axes[i].transAxes, verticalalignment='top')
    
    # Bos subplotlari kapat
    for i in range(len(numeric_cols), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path / "outlier_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    # Test icin basit kullanim
    import sys
    sys.path.append(".")
    from data_io import load_data
    
    # Ornek kullanim
    df = load_data(source="local", path="data/sample.csv")
    results = exploratory_data_analysis(df, target_col="value")