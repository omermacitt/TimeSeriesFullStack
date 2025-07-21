from __future__ import annotations
from pathlib import Path
import os
import pandas as pd
import joblib
import logging

logger = logging.getLogger(__name__)


def load_data(
        source: str = "local",
        fmt: str = "csv",
        path: str | Path | None = None,
        s3_uri: str | None = None,
        datetime_col: str = "timestamp",
        parse_dates: bool = True,
        sample_frac: float | None = None,
        cache: bool = True
):
    try:
        logger.info(f"Veri yükleme başlatıldı - kaynak: {source}, format: {fmt}")
        
        cache_file = Path("outputs/cache.pkl")
        if cache and cache_file.exists():
            logger.info("Cache'den veri yükleniyor")
            return joblib.load(cache_file)

        df = None
        
        if source == "local":
            if not path:
                logger.error("Local kaynak için path parametresi gerekli")
                raise ValueError("Local kaynak için path parametresi belirtilmeli")
                
            if not Path(path).exists():
                logger.error(f"Dosya bulunamadı: {path}")
                raise FileNotFoundError(f"Dosya bulunamadı: {path}")
            
            logger.info(f"Local dosya yükleniyor: {path}")
            if fmt == "csv":
                df = pd.read_csv(path, parse_dates=parse_dates)
            elif fmt == "parquet":
                df = pd.read_parquet(path)
            else:
                logger.error(f"Desteklenmeyen format: {fmt}")
                raise ValueError(f"Desteklenmeyen format: {fmt}")
        else:
            if not s3_uri:
                logger.error("S3 kaynağı için s3_uri parametresi gerekli")
                raise ValueError("S3 kaynağı için s3_uri parametresi belirtilmeli")
                
            try:
                import boto3, io
                logger.info(f"S3'den veri yükleniyor: {s3_uri}")
                s3 = boto3.client("s3")
                bucket, key = s3_uri.replace("s3://", "").split("/", 1)
                obj = s3.get_object(Bucket=bucket, Key=key)
                if fmt == "csv":
                    df = pd.read_csv(io.BytesIO(obj["Body"].read()), parse_dates=parse_dates)
                elif fmt == "parquet":
                    df = pd.read_parquet(io.BytesIO(obj["Body"].read()))
                else:
                    logger.error(f"Desteklenmeyen format: {fmt}")
                    raise ValueError(f"Desteklenmeyen format: {fmt}")
            except ImportError:
                logger.error("boto3 kütüphanesi bulunamadı. S3 kullanımı için boto3 yüklenmeli")
                raise ImportError("boto3 kütüphanesi bulunamadı")
            except Exception as e:
                logger.error(f"S3'den veri yüklenirken hata: {str(e)}")
                raise

        
        if df is None:
            logger.error("Veri yüklenemedi")
            raise ValueError("Veri yüklenemedi")
        
        logger.info(f"Veri başarıyla yüklendi. Shape: {df.shape}")

        if parse_dates:
            if datetime_col not in df.columns:
                logger.warning(f"Datetime sütunu bulunamadı: {datetime_col}")
            else:
                logger.info(f"Datetime sütunu işleniyor: {datetime_col}")
                df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")
                df = df.set_index(datetime_col).sort_index()

        logger.info("Veri tiplerini optimize ediliyor")
        for col, dtype in df.dtypes.items():
            if dtype == "float64":
                df[col] = pd.to_numeric(df[col], downcast="float")
            elif dtype == "int64":
                df[col] = pd.to_numeric(df[col], downcast="integer")

        if sample_frac:
            original_size = len(df)
            df = df.sample(frac=sample_frac)
            logger.info(f"Veri örneklendi: {original_size} -> {len(df)} satır")

        if cache:
            try:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(df, cache_file)
                logger.info(f"Veri cache'e kaydedildi: {cache_file}")
            except Exception as e:
                logger.warning(f"Cache kaydetme hatası: {str(e)}")

        logger.info("Veri yükleme işlemi tamamlandı")
        return df
        
    except Exception as e:
        logger.error(f"Veri yükleme sırasında beklenmeyen hata: {str(e)}")
        raise