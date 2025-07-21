"""
Plik z funkcjami pomocniczymi dla modułu treningowego.
"""
import logging
import os
import sys
from pathlib import Path

# Import config jako alias, aby uniknąć konfliktu nazw
from training2 import config as trainer_config

def find_project_root(marker_file=".project_root"):
    """
    Znajduje główny katalog projektu, szukając pliku-znacznika (.project_root).

    Funkcja przeszukuje drzewo katalogów w górę, zaczynając od lokalizacji 
    pliku, w którym jest wywoływana.

    Returns:
        Path: Obiekt Path wskazujący na główny katalog projektu.
    Raises:
        FileNotFoundError: Jeśli plik-znacznik nie zostanie znaleziony.
    """
    current_path = Path(__file__).resolve()
    while current_path != current_path.parent:
        if (current_path / marker_file).exists():
            return current_path
        current_path = current_path.parent
    raise FileNotFoundError(f"Nie znaleziono pliku znacznika '{marker_file}'. "
                            f"Upewnij się, że istnieje on w głównym katalogu projektu.")


def setup_logging():
    """Konfiguruje system logowania dla całego modułu."""
    log_dir = trainer_config.LOG_DIR
    os.makedirs(log_dir, exist_ok=True)
    
    log_filepath = os.path.join(log_dir, trainer_config.LOG_FILENAME)
    
    # Konfiguracja loggera
    logging.basicConfig(
        level=trainer_config.LOG_LEVEL,
        format=trainer_config.LOG_FORMAT,
        handlers=[
            logging.FileHandler(log_filepath),
            logging.StreamHandler(sys.stdout)
        ],
        force=True  # force=True jest ważne, jeśli funkcja może być wołana wielokrotnie
    )
    
    # Wyłącz zbyt gadatliwe loggery (jeśli będą problemy)
    # logging.getLogger('tensorflow').setLevel(logging.WARNING)
    
    return logging.getLogger(__name__) 