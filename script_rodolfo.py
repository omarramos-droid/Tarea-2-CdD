import subprocess
import sys
import importlib

def install_package(package_name):
    """Instala un paquete si no está disponible"""
    try:
        importlib.import_module(package_name)
        print(f"✅ {package_name} ya está instalado")
    except ImportError:
        print(f"📦 Instalando {package_name}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f"✅ {package_name} instalado correctamente")
        except subprocess.CalledProcessError:
            print(f"❌ Error instalando {package_name}")
            sys.exit(1)
            
            
if __name__ == "__main__":

    install_package("ucimlrepo")
