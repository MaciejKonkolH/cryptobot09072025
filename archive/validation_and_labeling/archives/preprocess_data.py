#!/usr/bin/env python3
"""
SKRYPT PREPROCESSING DANYCH - BATCH PROCESSING

Zgodnie z planem modyfikacji - MODUŁ 3: WORKFLOW PRODUKCYJNY
Wykorzystuje istniejący DataContinuityChecker do jednorazowego 
przetworzenia i zapisu czystych danych.

UŻYCIE:
    python scripts/preprocess_data.py --pair BTCUSDT
    python scripts/preprocess_data.py --input data/raw/ETHUSDT_1m.feather --output data/processed/ETHUSDT_1m_clean.feather

KORZYŚCI:
- Jednorazowe przetwarzanie zamiast przy każdym treningu
- Spójne dane dla wszystkich eksperymentów
- Szybsze uruchamianie treningów
- Backup czystych danych
"""

import argparse
import sys
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import os
import numpy as np  # Dodaję import numpy
import re  # Dodaję import re dla regex

# Dodaj ścieżkę do modułów projektu
sys.path.append(str(Path(__file__).parent.parent))

try:
    from core.utils.data_quality_validator import DataQualityValidator
except ImportError as e:
    print(f"❌ BŁĄD: Nie można zaimportować DataQualityValidator: {e}")
    print("   Sprawdź czy plik core/utils/data_quality_validator.py istnieje")
    sys.exit(1)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder dla wartości numpy."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def detect_timeframe_from_filename(filename: str) -> str:
    """
    Wykrywa timeframe z nazwy pliku.
    
    Args:
        filename: Nazwa pliku (np. "BTC_USDT-1m-futures.feather", "BTCUSDT_5m.feather")
        
    Returns:
        str: Wykryty timeframe (np. "1m", "5m", "1h", "1d")
    """
    # Wzorce timeframe w nazwach plików
    timeframe_patterns = [
        r'[-_](\d+m)[-_.]',     # 1m, 5m, 15m, 30m
        r'[-_](\d+h)[-_.]',     # 1h, 4h, 12h
        r'[-_](\d+d)[-_.]',     # 1d, 7d
        r'[-_](\d+w)[-_.]',     # 1w
        r'[-_](\d+M)[-_.]',     # 1M (miesięczne)
    ]
    
    filename_lower = filename.lower()
    
    for pattern in timeframe_patterns:
        match = re.search(pattern, filename_lower)
        if match:
            return match.group(1)
    
    # Jeśli nie znaleziono, sprawdź czy jest na końcu przed rozszerzeniem
    stem = Path(filename).stem.lower()
    
    # Wzorce na końcu nazwy
    end_patterns = [
        r'_(\d+[mhd])$',        # _1m, _5m, _1h, _1d
        r'-(\d+[mhd])$',        # -1m, -5m, -1h, -1d
    ]
    
    for pattern in end_patterns:
        match = re.search(pattern, stem)
        if match:
            return match.group(1)
    
    # Domyślny timeframe jeśli nie wykryto
    return "1m"


def extract_pair_from_filename(filename: str) -> str:
    """
    Wyciąga parę walutową z nazwy pliku.
    
    Args:
        filename: Nazwa pliku
        
    Returns:
        str: Para walutowa (np. "BTC_USDT", "BTCUSDT")
    """
    stem = Path(filename).stem
    
    # Usuń timeframe z nazwy
    timeframe = detect_timeframe_from_filename(filename)
    
    # Usuń timeframe i inne sufiksy
    clean_name = stem
    clean_name = re.sub(rf'[-_]{timeframe}[-_.].*$', '', clean_name)
    clean_name = re.sub(rf'[-_]{timeframe}$', '', clean_name)
    clean_name = re.sub(r'[-_]futures$', '', clean_name)
    clean_name = re.sub(r'[-_]spot$', '', clean_name)
    clean_name = re.sub(r'[-_]clean$', '', clean_name)
    clean_name = re.sub(r'[-_]validated$', '', clean_name)
    clean_name = re.sub(r'[-_]fixed$', '', clean_name)
    clean_name = re.sub(r'[-_]test$', '', clean_name)
    
    return clean_name.upper()


def generate_organized_paths(input_path: str, pair: str = None, timeframe: str = None) -> tuple:
    """
    Generuje zorganizowane ścieżki według wzorca {PARA}_{TIMEFRAME}.
    
    Args:
        input_path: Ścieżka do pliku wejściowego
        pair: Para walutowa (opcjonalne, wykryje automatycznie)
        timeframe: Timeframe (opcjonalne, wykryje automatycznie)
        
    Returns:
        tuple: (output_path, report_path, folder_name)
    """
    if not pair:
        pair = extract_pair_from_filename(input_path)
    
    if not timeframe:
        timeframe = detect_timeframe_from_filename(input_path)
    
    # Nazwa folderu według wzorca {PARA}_{TIMEFRAME}
    folder_name = f"{pair}_{timeframe}"
    
    # Ścieżki wyjściowe
    base_dir = Path("data/validated")
    folder_path = base_dir / folder_name
    
    # Nazwy plików
    input_filename = Path(input_path).stem
    output_filename = f"{pair}_{timeframe}_validated.feather"
    report_filename = f"{pair}_{timeframe}_quality_report.json"
    
    output_path = folder_path / output_filename
    report_path = folder_path / report_filename
    
    return str(output_path), str(report_path), folder_name


def preprocess_and_save_data(
    input_path: str,
    output_path: str,
    report_path: str,
    min_quality: float = 80.0,
    # NOWE PARAMETRY WALIDACJI JAKOŚCI:
    enable_quality_validation: bool = True,  # Domyślnie włączone - pełna walidacja
    quality_checks: list = None,
    anomaly_thresholds: dict = None,
    statistical_thresholds: dict = None,
    # NOWE PARAMETRY COMPETITIVE LABELING:
    enable_competitive_labeling: bool = False,  # Competitive labeling
    competitive_config: dict = None
) -> dict:
    """
    Jednorazowe przetwarzanie i zapis danych z domyślnie włączoną walidacją jakości 
    i opcjonalnym competitive labeling.
    
    Args:
        input_path: Ścieżka do surowych danych
        output_path: Ścieżka zapisu czystych danych  
        report_path: Ścieżka zapisu raportu JSON
        min_quality: Minimalny próg jakości (%)
        enable_quality_validation: Włącz walidację jakości danych (domyślnie: True)
        quality_checks: Lista sprawdzeń jakości do wykonania
        anomaly_thresholds: Progi dla wykrywania anomalii cenowych
        statistical_thresholds: Progi dla analizy statystycznej
        enable_competitive_labeling: Włącz labeling konkurencyjne (domyślnie: False)
        competitive_config: Konfiguracja labeling konkurencyjnego (TP/SL/WINDOW)
        
    Returns:
        dict: Statystyki preprocessing
    """
    print(f"🔄 PREPROCESSING DANYCH")
    print(f"   Wejście: {input_path}")
    print(f"   Wyjście: {output_path}")
    print(f"   Raport: {report_path}")
    print(f"   Min. jakość: {min_quality}%")
    if enable_quality_validation:
        print(f"   Walidacja jakości: WŁĄCZONA")
        print(f"   Sprawdzenia: {', '.join(quality_checks) if quality_checks else 'wszystkie'}")
    else:
        print(f"   Walidacja jakości: wyłączona (tylko ciągłość)")
    
    if enable_competitive_labeling:
        print(f"   Competitive labeling: WŁĄCZONY")
        if competitive_config:
            print(f"   Parametry TP/SL: LONG {competitive_config.get('LONG_TP_PCT', 0.01)*100:.1f}%/{competitive_config.get('LONG_SL_PCT', 0.005)*100:.1f}%, SHORT {competitive_config.get('SHORT_TP_PCT', 0.01)*100:.1f}%/{competitive_config.get('SHORT_SL_PCT', 0.005)*100:.1f}%")
            print(f"   Future window: {competitive_config.get('FUTURE_WINDOW', 120)} minut")
        else:
            print(f"   Parametry: domyślne (TP 1.0%, SL 0.5%, okno 120min)")
    else:
        print(f"   Competitive labeling: wyłączony")
    print("-" * 50)
    
    # 1. Sprawdź czy plik wejściowy istnieje
    if not Path(input_path).exists():
        raise FileNotFoundError(f"Plik wejściowy nie istnieje: {input_path}")
    
    # 2. Utwórz katalogi wyjściowe jeśli nie istnieją
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    
    # 3. Załaduj surowe dane
    print("📂 Ładowanie surowych danych...")
    try:
        if input_path.endswith('.feather'):
            df_raw = pd.read_feather(input_path)
            # Sprawdź czy jest kolumna timestamp i ustaw jako index
            if 'timestamp' in df_raw.columns:
                df_raw = df_raw.set_index('timestamp')
                # Sprawdź czy to Unix timestamp w milisekundach
                if df_raw.index.dtype in ['int64', 'float64']:
                    # Unix timestamp w milisekundach - konwertuj poprawnie
                    df_raw.index = pd.to_datetime(df_raw.index, unit='ms')
                else:
                    df_raw.index = pd.to_datetime(df_raw.index)
            elif 'index' in df_raw.columns:
                # Kolumna 'index' zawiera timestamp
                df_raw = df_raw.set_index('index')
                # Sprawdź czy to Unix timestamp w milisekundach
                if df_raw.index.dtype in ['int64', 'float64']:
                    df_raw.index = pd.to_datetime(df_raw.index, unit='ms')
                else:
                    df_raw.index = pd.to_datetime(df_raw.index)
            elif df_raw.index.name == 'timestamp' or isinstance(df_raw.index, pd.DatetimeIndex):
                # Index już jest ustawiony poprawnie
                pass
            else:
                # Sprawdź czy pierwsza kolumna to timestamp
                if df_raw.columns[0] in ['timestamp', 'date', 'datetime', 'index']:
                    df_raw = df_raw.set_index(df_raw.columns[0])
                    # Sprawdź czy to Unix timestamp w milisekundach
                    if df_raw.index.dtype in ['int64', 'float64']:
                        df_raw.index = pd.to_datetime(df_raw.index, unit='ms')
                    else:
                        df_raw.index = pd.to_datetime(df_raw.index)
                else:
                    raise ValueError("Nie można znaleźć kolumny timestamp w danych feather")
        elif input_path.endswith('.csv'):
            df_raw = pd.read_csv(input_path, index_col=0, parse_dates=True)
        elif input_path.endswith('.parquet'):
            df_raw = pd.read_parquet(input_path)
        else:
            raise ValueError(f"Nieobsługiwany format pliku: {input_path}")
            
        print(f"   ✅ Załadowano {len(df_raw)} świec")
        print(f"   📅 Zakres: {df_raw.index[0]} → {df_raw.index[-1]}")
        
    except Exception as e:
        raise RuntimeError(f"Błąd ładowania danych: {e}")
    
    # 4. Sprawdź i uzupełnij luki używając DataContinuityChecker
    print("🔍 Sprawdzanie ciągłości danych...")
    
    # Wykryj timeframe z nazwy pliku
    timeframe = detect_timeframe_from_filename(input_path)
    print(f"   Wykryto timeframe: {timeframe}")
    
    # Utwórz validator z wszystkimi parametrami
    validator = DataQualityValidator(
        timeframe=timeframe,
        tolerance_seconds=60,
        enable_quality_validation=enable_quality_validation,
        quality_checks=quality_checks,
        anomaly_thresholds=anomaly_thresholds,
        statistical_thresholds=statistical_thresholds,
        enable_competitive_labeling=enable_competitive_labeling,
        competitive_config=competitive_config
    )
    
    # Sprawdź i napraw dane
    df_clean, report = validator.check_and_fill_gaps(df_raw)
    
    # 5. Oceń jakość danych
    quality_score = report['quality_score']
    comprehensive_score = report.get('comprehensive_quality_score')
    
    print(f"📊 WYNIKI ANALIZY:")
    print(f"   Oryginalne świece: {report['original_candles']}")
    print(f"   Wykryte luki: {report['gaps_detected']}")
    print(f"   Dodane świece: {report['candles_added']}")
    print(f"   Finalne świece: {len(df_clean)}")
    print(f"   Jakość danych (ciągłość): {quality_score:.1f}%")
    
    if enable_quality_validation and comprehensive_score is not None:
        print(f"   Kompleksowa jakość: {comprehensive_score:.1f}%")
        print(f"   Status jakości: {report.get('comprehensive_quality_status', 'N/A')}")
        
        # Pokaż breakdown jeśli dostępny
        if 'quality_breakdown' in report:
            breakdown = report['quality_breakdown']
            print(f"   Breakdown jakości:")
            print(f"     - Ciągłość: {breakdown['continuity_score']:.1f}%")
            if breakdown.get('ohlcv_logic_score') is not None:
                print(f"     - OHLCV Logic: {breakdown['ohlcv_logic_score']:.1f}%")
            if breakdown.get('anomaly_score') is not None:
                print(f"     - Anomalie: {breakdown['anomaly_score']:.1f}%")
            if breakdown.get('statistical_score') is not None:
                print(f"     - Statystyki: {breakdown['statistical_score']:.1f}%")
        
        # Pokaż podsumowanie problemów
        if 'validation_summary' in report:
            summary = report['validation_summary']
            if summary['total_issues'] > 0:
                print(f"   Problemy z jakością:")
                print(f"     - Łącznie: {summary['total_issues']}")
                print(f"     - Krytyczne: {summary['critical_issues']}")
                print(f"     - Ostrzeżenia: {summary['warnings']}")
        
        # Użyj comprehensive score do oceny progu
        final_score = comprehensive_score
    else:
        final_score = quality_score
    
    # 6. Sprawdź próg jakości
    if final_score < min_quality:
        print(f"⚠️  OSTRZEŻENIE: Jakość danych ({final_score:.1f}%) poniżej progu ({min_quality}%)")
        print("   Dane zostaną zapisane, ale mogą wymagać uwagi")
    else:
        print(f"✅ Jakość danych powyżej progu - dane gotowe do użycia")
    
    # 7. Zapisz czyste dane
    print("💾 Zapisywanie czystych danych...")
    try:
        if output_path.endswith('.feather'):
            df_clean.reset_index().to_feather(output_path)
        elif output_path.endswith('.csv'):
            df_clean.to_csv(output_path)
        elif output_path.endswith('.parquet'):
            df_clean.to_parquet(output_path)
        else:
            # Domyślnie feather
            output_path = output_path.replace('.', '_clean.feather')
            df_clean.reset_index().to_feather(output_path)
            
        print(f"   ✅ Zapisano: {output_path}")
        
    except Exception as e:
        raise RuntimeError(f"Błąd zapisu danych: {e}")
    
    # 8. Przygotuj rozszerzony raport
    extended_report = {
        **report,
        'preprocessing_info': {
            'timestamp': datetime.now().isoformat(),
            'input_path': input_path,
            'output_path': output_path,
            'min_quality_threshold': min_quality,
            'quality_passed': final_score >= min_quality,
            'file_size_mb': round(Path(output_path).stat().st_size / 1024 / 1024, 2)
        }
    }
    
    # 9. Zapisz raport JSON
    print("📋 Zapisywanie raportu...")
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(extended_report, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        print(f"   ✅ Zapisano: {report_path}")
        
    except Exception as e:
        print(f"   ⚠️  Błąd zapisu raportu: {e}")
    
    # 10. Podsumowanie
    print(f"\n✅ PREPROCESSING ZAKOŃCZONY")
    print(f"   Status: {'✅ PASSED' if final_score >= min_quality else '⚠️ BELOW THRESHOLD'}")
    print(f"   Jakość: {final_score:.1f}%")
    print(f"   Pliki zapisane:")
    print(f"     Dane: {output_path}")
    print(f"     Raport: {report_path}")
    
    # Dodatkowe informacje o competitive labeling
    if 'competitive_labeling' in report and report['competitive_labeling'].get('enabled'):
        labeling_info = report['competitive_labeling']
        if labeling_info.get('labels_generated', 0) > 0:
            print(f"   Competitive labeling:")
            print(f"     Etykiety: {labeling_info['labels_generated']:,}")
            for label, count in labeling_info['label_distribution'].items():
                percentage = labeling_info['label_percentages'][label]
                print(f"     {label}: {count:,} ({percentage:.1f}%)")
    
    # Zwróć rozszerzone statystyki
    stats = {
        'quality_score': final_score,
        'quality_passed': final_score >= min_quality,
        'input_candles': report['original_candles'],
        'final_candles': len(df_clean),
        'added_candles': report['candles_added'],
        'gaps_detected': report['gaps_detected'],
        'processing_time': datetime.now().isoformat(),
        'enable_quality_validation': enable_quality_validation,
        'enable_competitive_labeling': enable_competitive_labeling
    }
    
    # Dodaj statystyki competitive labeling jeśli dostępne
    if 'competitive_labeling' in report and report['competitive_labeling'].get('enabled'):
        labeling_info = report['competitive_labeling']
        stats['competitive_labeling'] = {
            'labels_generated': labeling_info.get('labels_generated', 0),
            'label_distribution': labeling_info.get('label_distribution', {}),
            'parameters': labeling_info.get('labeling_parameters', {})
        }
    
    return stats


def main():
    """Główna funkcja skryptu"""
    parser = argparse.ArgumentParser(
        description="Preprocessing danych historycznych - uzupełnianie luk",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
PRZYKŁADY UŻYCIA:

  # Podstawowe użycie - automatyczna organizacja folderów
  python scripts/preprocess_data.py --pair BTCUSDT
  # Utworzy: data/validated/BTC_USDT_1m/BTC_USDT_1m_validated.feather
  #          data/validated/BTC_USDT_1m/BTC_USDT_1m_quality_report.json
  
  # Z określonym timeframe
  python scripts/preprocess_data.py --pair BTCUSDT --timeframe 5m
  # Utworzy: data/validated/BTC_USDT_5m/BTC_USDT_5m_validated.feather
  
  # Tylko ciągłość danych (bez walidacji jakości)
  python scripts/preprocess_data.py --pair BTCUSDT --disable-quality-validation
  
  # Automatyczna organizacja z pliku wejściowego
  python scripts/preprocess_data.py --input "ft_bot_docker_compose/user_data/data/binanceusdm/futures/BTC_USDT-1m-futures.feather"
  # Wykryje: para=BTC_USDT, timeframe=1m
  # Utworzy: data/validated/BTC_USDT_1m/BTC_USDT_1m_validated.feather
  
  # Z niestandardowymi progami (pełna walidacja)
  python scripts/preprocess_data.py --pair BTCUSDT \\
    --price-jump-threshold 3.0 \\
    --outlier-sigma 2.5 \\
    --min-quality 85
    
  # Ręczne ścieżki (stary sposób)
  python scripts/preprocess_data.py \\
    --input data/raw/ETHUSDT_1m.feather \\
    --output data/custom/ETHUSDT_clean.feather \\
    --report data/custom/ETHUSDT_report.json
    
  # Tylko wybrane sprawdzenia jakości
  python scripts/preprocess_data.py --pair ADAUSDT \\
    --quality-checks price_anomalies \\
    --price-jump-threshold 2.0
    
  # Szybkie przetwarzanie bez walidacji jakości
  python scripts/preprocess_data.py --pair BTCUSDT \\
    --disable-quality-validation \\
    --min-quality 70

STRUKTURA FOLDERÓW:
  data/validated/
  ├── BTC_USDT_1m/
  │   ├── BTC_USDT_1m_validated.feather
  │   └── BTC_USDT_1m_quality_report.json
  ├── BTC_USDT_5m/
  │   ├── BTC_USDT_5m_validated.feather
  │   └── BTC_USDT_5m_quality_report.json
  └── ETH_USDT_1h/
      ├── ETH_USDT_1h_validated.feather
      └── ETH_USDT_1h_quality_report.json
        """
    )
    
    # Grupa argumentów - albo --pair albo ręczne ścieżki
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--pair',
        type=str,
        help='Para walutowa (np. BTCUSDT) - automatyczne ścieżki'
    )
    group.add_argument(
        '--input',
        type=str,
        help='Ścieżka do surowych danych'
    )
    
    # Opcjonalne argumenty
    parser.add_argument(
        '--timeframe',
        type=str,
        help='Timeframe (np. 1m, 5m, 1h, 1d) - wykryje automatycznie jeśli nie podano'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Ścieżka zapisu czystych danych (opcjonalne - automatyczne jeśli nie podano)'
    )
    parser.add_argument(
        '--report',
        type=str,
        help='Ścieżka zapisu raportu JSON (opcjonalne - automatyczne jeśli nie podano)'
    )
    parser.add_argument(
        '--min-quality',
        type=float,
        default=80.0,
        help='Minimalny próg jakości danych w %% (domyślnie: 80.0)'
    )
    
    # NOWE ARGUMENTY WALIDACJI JAKOŚCI
    parser.add_argument(
        '--disable-quality-validation',
        action='store_true',
        help='Wyłącz walidację jakości danych (domyślnie: włączona)'
    )
    parser.add_argument(
        '--quality-checks',
        type=str,
        help='Lista sprawdzeń jakości oddzielona przecinkami (ohlcv_logic,price_anomalies,statistical_patterns)'
    )
    parser.add_argument(
        '--price-jump-threshold',
        type=float,
        help='Próg dla skoków cenowych w %% (domyślnie: 5.0)'
    )
    parser.add_argument(
        '--outlier-sigma',
        type=float,
        help='Próg dla outliers w sigma (domyślnie: 3.0)'
    )
    parser.add_argument(
        '--extreme-spread-threshold',
        type=float,
        help='Próg dla ekstremalnych spreadów w %% (domyślnie: 20.0)'
    )
    parser.add_argument(
        '--flash-crash-threshold',
        type=float,
        help='Próg dla flash crashes w %% (domyślnie: -10.0)'
    )
    parser.add_argument(
        '--uniqueness-min',
        type=float,
        default=0.1,
        help='Minimalny próg unikalności (domyślnie: 0.1)'
    )
    parser.add_argument(
        '--volatility-max',
        type=float,
        default=0.1,
        help='Maksymalny próg volatility (domyślnie: 0.1)'
    )
    
    # PARAMETRY COMPETITIVE LABELING
    parser.add_argument('--enable-competitive-labeling', action='store_true',
                       help='Włącz competitive labeling (etykietowanie danych)')
    parser.add_argument('--long-tp', type=float, default=1.0,
                       help='LONG Take Profit w procentach (domyślnie: 1.0)')
    parser.add_argument('--long-sl', type=float, default=0.5,
                       help='LONG Stop Loss w procentach (domyślnie: 0.5)')
    parser.add_argument('--short-tp', type=float, default=1.0,
                       help='SHORT Take Profit w procentach (domyślnie: 1.0)')
    parser.add_argument('--short-sl', type=float, default=0.5,
                       help='SHORT Stop Loss w procentach (domyślnie: 0.5)')
    parser.add_argument('--future-window', type=int, default=120,
                       help='Okno przyszłości w minutach (domyślnie: 120)')
    
    args = parser.parse_args()
    
    # Walidacja argumentów - teraz --output nie jest wymagane
    # Automatycznie wygenerujemy ścieżki jeśli nie podano
    
    # Określ ścieżki
    if args.pair:
        # Automatyczne ścieżki na podstawie pary
        pair = args.pair.upper()
        timeframe = args.timeframe or "1m"  # Domyślny timeframe
        
        # Znajdź plik wejściowy - sprawdź różne lokalizacje
        possible_inputs = [
            f"data/raw/{pair}_{timeframe}.feather",
            f"data/raw/{pair}-{timeframe}.feather", 
            f"data/processed/{pair}_{timeframe}.feather",
            f"data/processed/{pair}-{timeframe}.feather",
            f"ft_bot_docker_compose/user_data/data/binanceusdm/futures/{pair}-{timeframe}-futures.feather",
            f"ft_bot_docker_compose/user_data/data/binanceusdm/futures/{pair}_{timeframe}_futures.feather"
        ]
        
        input_path = None
        for possible_path in possible_inputs:
            if Path(possible_path).exists():
                input_path = possible_path
                break
        
        if not input_path:
            print(f"❌ BŁĄD: Nie znaleziono pliku dla pary {pair} i timeframe {timeframe}")
            print("   Sprawdzone lokalizacje:")
            for path in possible_inputs:
                print(f"     - {path}")
            sys.exit(1)
        
        # Wygeneruj zorganizowane ścieżki
        output_path, report_path, folder_name = generate_organized_paths(input_path, pair, timeframe)
        
    else:
        # Ręczne ścieżki
        input_path = args.input
        
        if not Path(input_path).exists():
            print(f"❌ BŁĄD: Plik wejściowy nie istnieje: {input_path}")
            sys.exit(1)
        
        # Jeśli nie podano output/report, wygeneruj automatycznie
        if args.output or args.report:
            # Użytkownik podał własne ścieżki
            output_path = args.output
            report_path = args.report
            folder_name = "custom"
            
            if not output_path:
                # Wygeneruj output na podstawie input
                output_path, report_path, folder_name = generate_organized_paths(
                    input_path, timeframe=args.timeframe
                )
            elif not report_path:
                # Wygeneruj tylko report
                _, report_path, _ = generate_organized_paths(
                    input_path, timeframe=args.timeframe
                )
        else:
            # Automatyczne ścieżki
            output_path, report_path, folder_name = generate_organized_paths(
                input_path, timeframe=args.timeframe
            )
    
    # Wyświetl informacje o organizacji
    print(f"📁 ORGANIZACJA PLIKÓW:")
    print(f"   Folder: {folder_name}")
    print(f"   Wejście: {input_path}")
    print(f"   Wyjście: {output_path}")
    print(f"   Raport: {report_path}")
    print()
    
    try:
        # NOWE: Przygotuj parametry walidacji jakości
        quality_checks = None
        if args.quality_checks:
            quality_checks = [check.strip() for check in args.quality_checks.split(',')]
            # Waliduj nazwy sprawdzeń
            valid_checks = ['ohlcv_logic', 'price_anomalies', 'statistical_patterns']
            invalid_checks = [check for check in quality_checks if check not in valid_checks]
            if invalid_checks:
                parser.error(f"Nieprawidłowe sprawdzenia jakości: {invalid_checks}. Dostępne: {valid_checks}")
        
        # Przygotuj progi anomalii
        anomaly_thresholds = {}
        if args.price_jump_threshold is not None:
            anomaly_thresholds['price_jump'] = args.price_jump_threshold / 100.0  # Konwersja % na ułamek
        if args.outlier_sigma is not None:
            anomaly_thresholds['outlier_sigma'] = args.outlier_sigma
        if args.extreme_spread_threshold is not None:
            anomaly_thresholds['extreme_spread'] = args.extreme_spread_threshold / 100.0
        if args.flash_crash_threshold is not None:
            anomaly_thresholds['flash_crash'] = args.flash_crash_threshold / 100.0
        
        # Konfiguracja progów statystycznych
        statistical_thresholds = {
            'uniqueness_min': args.uniqueness_min,
            'volatility_max': args.volatility_max
        }
        
        # NOWA: Konfiguracja competitive labeling
        competitive_config = None
        if args.enable_competitive_labeling:
            competitive_config = {
                'LONG_TP_PCT': args.long_tp / 100.0,   # Konwersja % na decimal
                'LONG_SL_PCT': args.long_sl / 100.0,   # Konwersja % na decimal
                'SHORT_TP_PCT': args.short_tp / 100.0, # Konwersja % na decimal
                'SHORT_SL_PCT': args.short_sl / 100.0, # Konwersja % na decimal
                'FUTURE_WINDOW': args.future_window
            }
        
        print(f"🚀 Uruchamianie preprocessing...")
        
        # Wykonaj preprocessing
        stats = preprocess_and_save_data(
            input_path=input_path,
            output_path=output_path,
            report_path=report_path,
            min_quality=args.min_quality,
            enable_quality_validation=not args.disable_quality_validation,
            quality_checks=quality_checks,
            anomaly_thresholds=anomaly_thresholds,
            statistical_thresholds=statistical_thresholds,
            enable_competitive_labeling=args.enable_competitive_labeling,
            competitive_config=competitive_config
        )
        
        # Podsumowanie
        print(f"\n📈 STATYSTYKI KOŃCOWE:")
        print(f"   Wejście: {stats['input_candles']:,} świec")
        print(f"   Wyjście: {stats['final_candles']:,} świec")
        print(f"   Dodane: {stats['added_candles']:,} świec")
        print(f"   Luki: {stats['gaps_detected']}")
        print(f"   Jakość: {stats['quality_score']:.1f}%")
        print(f"   Status: {'✅ PASSED' if stats['quality_passed'] else '⚠️ BELOW THRESHOLD'}")
        
        if not stats['quality_passed']:
            print(f"\n⚠️  UWAGA: Jakość danych poniżej progu {args.min_quality}%")
            print("   Sprawdź raport i rozważ użycie danych z ostrożnością")
            sys.exit(2)  # Exit code 2 = warning
        
    except Exception as e:
        print(f"\n❌ BŁĄD PREPROCESSING: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 