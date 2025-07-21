from __future__ import annotations
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
import logging

logger = logging.getLogger(__name__)

def quick_quality_check(df: pd.DataFrame, output_path: str = "outputs/qc_baseline.json") -> dict:
    """
    Hizli kalite kontrolu - kritik sorunlari 15-20 dakikada tespit et
    """
    console = Console()
    console.print("\n[bold blue]📊 Hizli Kalite Kontrolu Baslatiliyor...[/bold blue]")
    
    try:
        # QC sonuclari icin temel yapi
        qc_results = {
            "timestamp": datetime.now().isoformat(),
            "dataset_info": {},
            "missing_values": {},
            "outliers": {},
            "data_types": {},
            "summary_stats": {},
            "quality_score": 0,
            "critical_issues": []
        }
        
        # 1. Temel dataset bilgileri
        logger.info("Dataset temel bilgileri toplaniyor")
        qc_results["dataset_info"] = {
            "shape": df.shape,
            "columns": list(df.columns),
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
            "index_type": str(type(df.index)),
            "date_range": {
                "start": str(df.index.min()) if hasattr(df.index, 'min') else None,
                "end": str(df.index.max()) if hasattr(df.index, 'max') else None
            } if pd.api.types.is_datetime64_any_dtype(df.index) else None
        }
        
        # 2. Eksik deger analizi
        logger.info("Eksik deger analizi yapiliyor")
        missing_analysis = {}
        for col in df.columns:
            missing_count = df[col].isna().sum()
            missing_pct = (missing_count / len(df)) * 100
            missing_analysis[col] = {
                "count": int(missing_count),
                "percentage": round(missing_pct, 2),
                "critical": missing_pct > 50  # %50'den fazla eksik kritik
            }
        qc_results["missing_values"] = missing_analysis
        
        # 3. Aykiri deger analizi (sadece numerik sutunlar)
        logger.info("Aykiri deger analizi yapiliyor")
        outlier_analysis = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df[col].notna().sum() > 0:  # Sadece veri varsa analiz et
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
                outlier_pct = (len(outliers) / len(df[col].dropna())) * 100
                
                outlier_analysis[col] = {
                    "count": len(outliers),
                    "percentage": round(outlier_pct, 2),
                    "bounds": {"lower": float(lower_bound), "upper": float(upper_bound)},
                    "critical": outlier_pct > 10  # %10'dan fazla aykiri kritik
                }
        qc_results["outliers"] = outlier_analysis
        
        # 4. Veri tipi kontrolu
        logger.info("Veri tipi analizi yapiliyor")
        dtype_analysis = {}
        for col in df.columns:
            dtype_analysis[col] = {
                "current_type": str(df[col].dtype),
                "null_count": int(df[col].isna().sum()),
                "unique_values": int(df[col].nunique()),
                "memory_usage_kb": round(df[col].memory_usage(deep=True) / 1024, 2)
            }
        qc_results["data_types"] = dtype_analysis
        
        # 5. Ozet istatistikler
        logger.info("Ozet istatistikler hesaplaniyor")
        summary_stats = {}
        for col in numeric_cols:
            if df[col].notna().sum() > 0:
                summary_stats[col] = {
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "skewness": float(df[col].skew()),
                    "kurtosis": float(df[col].kurtosis())
                }
        qc_results["summary_stats"] = summary_stats
        
        # 6. Kritik sorunlari belirle
        critical_issues = []
        
        # Yuksek eksik deger orani
        for col, info in missing_analysis.items():
            if info["critical"]:
                critical_issues.append(f"Yuksek eksik deger: {col} (%{info['percentage']})")
        
        # Yuksek aykiri deger orani
        for col, info in outlier_analysis.items():
            if info["critical"]:
                critical_issues.append(f"Yuksek aykiri deger: {col} (%{info['percentage']})")
        
        # Sifir varyans
        for col in numeric_cols:
            if df[col].std() == 0:
                critical_issues.append(f"Sifir varyans: {col}")
        
        qc_results["critical_issues"] = critical_issues
        
        # 7. Kalite skoru hesapla (0-100)
        quality_score = 100
        quality_score -= len(critical_issues) * 10  # Her kritik sorun -10 puan
        quality_score -= sum(1 for info in missing_analysis.values() if info["percentage"] > 10) * 5  # %10+ eksik -5 puan
        quality_score = max(0, quality_score)  # Minimum 0
        qc_results["quality_score"] = quality_score
        
        # 8. Sonuclari dosyaya kaydet
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(qc_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"QC sonuclari kaydedildi: {output_path}")
        
        # 9. Terminal ciktisi olustur
        _display_qc_results(console, qc_results, df)
        
        return qc_results
        
    except Exception as e:
        logger.error(f"QC analizi sirasinda hata: {str(e)}")
        raise


def _display_qc_results(console: Console, qc_results: dict, df: pd.DataFrame):
    """Terminal'de renkli QC sonuclarini goster"""
    
    # Genel Bilgiler
    info_table = Table(title="📊 Dataset Genel Bilgileri", box=box.ROUNDED)
    info_table.add_column("Ozellik", style="cyan")
    info_table.add_column("Deger", style="white")
    
    shape = qc_results["dataset_info"]["shape"]
    info_table.add_row("Boyut", f"{shape[0]:,} satir × {shape[1]} sutun")
    info_table.add_row("Bellek Kullanimi", f"{qc_results['dataset_info']['memory_usage_mb']} MB")
    info_table.add_row("Kalite Skoru", f"[bold green]{qc_results['quality_score']}/100[/bold green]")
    
    if qc_results["dataset_info"]["date_range"]:
        date_range = qc_results["dataset_info"]["date_range"]
        info_table.add_row("Tarih Araligi", f"{date_range['start']} - {date_range['end']}")
    
    console.print(info_table)
    console.print()
    
    # Eksik Degerler Tablosu
    if qc_results["missing_values"]:
        missing_table = Table(title="❌ Eksik Deger Analizi", box=box.ROUNDED)
        missing_table.add_column("Sutun", style="cyan")
        missing_table.add_column("Eksik Sayisi", justify="right")
        missing_table.add_column("Eksik %", justify="right")
        missing_table.add_column("Durum", justify="center")
        
        for col, info in qc_results["missing_values"].items():
            status = "[red]KRITIK[/red]" if info["critical"] else "[yellow]DIKKAT[/yellow]" if info["percentage"] > 10 else "[green]IYI[/green]"
            missing_table.add_row(
                col, 
                f"{info['count']:,}", 
                f"{info['percentage']:.1f}%", 
                status
            )
        
        console.print(missing_table)
        console.print()
    
    # Aykiri Degerler Tablosu
    if qc_results["outliers"]:
        outlier_table = Table(title="📈 Aykiri Deger Analizi", box=box.ROUNDED)
        outlier_table.add_column("Sutun", style="cyan")
        outlier_table.add_column("Aykiri Sayisi", justify="right")
        outlier_table.add_column("Aykiri %", justify="right")
        outlier_table.add_column("Durum", justify="center")
        
        for col, info in qc_results["outliers"].items():
            status = "[red]KRITIK[/red]" if info["critical"] else "[yellow]DIKKAT[/yellow]" if info["percentage"] > 5 else "[green]IYI[/green]"
            outlier_table.add_row(
                col, 
                f"{info['count']:,}", 
                f"{info['percentage']:.1f}%", 
                status
            )
        
        console.print(outlier_table)
        console.print()
    
    # Veri Tipleri Tablosu
    dtype_table = Table(title="🔢 Veri Tipi Analizi", box=box.ROUNDED)
    dtype_table.add_column("Sutun", style="cyan")
    dtype_table.add_column("Tip", style="white")
    dtype_table.add_column("Unique", justify="right")
    dtype_table.add_column("Bellek (KB)", justify="right")
    
    for col, info in qc_results["data_types"].items():
        dtype_table.add_row(
            col,
            info["current_type"],
            f"{info['unique_values']:,}",
            f"{info['memory_usage_kb']:.1f}"
        )
    
    console.print(dtype_table)
    console.print()
    
    # Kritik Sorunlar
    if qc_results["critical_issues"]:
        issues_text = "\n".join([f"• {issue}" for issue in qc_results["critical_issues"]])
        console.print(Panel(
            issues_text,
            title="🚨 Kritik Sorunlar",
            border_style="red",
            expand=False
        ))
    else:
        console.print(Panel(
            "Kritik sorun bulunamadi! ✅",
            title="🎉 Kalite Kontrolu",
            border_style="green",
            expand=False
        ))
    
    console.print(f"\n[bold green]✅ QC analizi tamamlandi! Sonuclar outputs/qc_baseline.json'a kaydedildi.[/bold green]")


if __name__ == "__main__":
    # Test icin basit kullanim
    import sys
    sys.path.append(".")
    from data_io import load_data
    
    # Ornek kullanim
    df = load_data(source="local", path="data/sample.csv")  # Ornek path
    results = quick_quality_check(df)