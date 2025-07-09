import subprocess
import sys
import os

def get_installed_packages():
    """
    Używa aktywnego interpretera Python do uruchomienia 'pip freeze'
    i zwraca listę zainstalowanych pakietów.
    """
    try:
        # Użyj sys.executable, aby mieć pewność, że korzystamy z pip
        # z tego samego środowiska, w którym uruchomiony jest skrypt.
        python_executable = sys.executable
        command = [python_executable, "-m", "pip", "freeze"]
        
        print(f"🚀 Uruchamiam komendę: {' '.join(command)}")
        
        # Uruchom komendę i przechwyć jej wynik
        result = subprocess.run(
            command, 
            capture_output=True, 
            text=True, 
            check=True,
            encoding='utf-8' # Jawne określenie kodowania
        )
        
        return result.stdout
        
    except FileNotFoundError:
        print("❌ BŁĄD: 'pip' nie został znaleziony. Upewnij się, że Python i pip są w PATH.")
        return None
    except subprocess.CalledProcessError as e:
        print(f"❌ BŁĄD podczas uruchamiania 'pip freeze':\n{e.stderr}")
        return None

def save_to_file(content, filename="runpod_requirements.txt"):
    """Zapisuje zawartość do pliku."""
    try:
        with open(filename, "w", encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Pomyślnie zapisano zależności do pliku: {os.path.abspath(filename)}")
    except IOError as e:
        print(f"❌ BŁĄD podczas zapisywania do pliku {filename}:\n{e}")

if __name__ == "__main__":
    print("🤖 Rozpoczynam generowanie listy zainstalowanych pakietów na RunPod...")
    
    packages = get_installed_packages()
    
    if packages:
        save_to_file(packages)
    else:
        print("🔴 Nie udało się wygenerować listy pakietów.")
