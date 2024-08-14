import subprocess
import pkg_resources
import sys

# Korumak istediğiniz paketler
keep_packages = {'pip', 'setuptools', 'wheel', 'virtualenv'}

# Tüm yüklü paketleri listele
installed_packages = {pkg.key for pkg in pkg_resources.working_set}

# Kaldırılacak paketleri belirle
packages_to_remove = installed_packages - keep_packages

# Paketleri kaldır
for package in packages_to_remove:
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", package])

print("Temizlik işlemi tamamlandı.")