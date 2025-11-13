"""
Utility functions for NAS training
"""
import os
import sys
import warnings
from typing import Any

# Suppress MNE warnings
os.environ['MNE_LOGGING_LEVEL'] = 'ERROR'
warnings.filterwarnings('ignore', category=RuntimeWarning, module='mne')
warnings.filterwarnings('ignore', category=UserWarning, module='mne')

# Try to disable MNE verbose output
try:
    import mne
    mne.set_log_level('ERROR')
except:
    pass


class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
    @staticmethod
    def disable():
        """Disable colors"""
        Colors.HEADER = ''
        Colors.OKBLUE = ''
        Colors.OKCYAN = ''
        Colors.OKGREEN = ''
        Colors.WARNING = ''
        Colors.FAIL = ''
        Colors.ENDC = ''
        Colors.BOLD = ''
        Colors.UNDERLINE = ''


def print_header(text: str):
    """Print a header with colored formatting"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text:^80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 80}{Colors.ENDC}\n")


def print_section(text: str):
    """Print a section header"""
    print(f"\n{Colors.OKCYAN}{Colors.BOLD}{'─' * 60}{Colors.ENDC}")
    print(f"{Colors.OKCYAN}{Colors.BOLD}{text}{Colors.ENDC}")
    print(f"{Colors.OKCYAN}{Colors.BOLD}{'─' * 60}{Colors.ENDC}")


def print_info(text: str):
    """Print informational message"""
    print(f"{Colors.OKBLUE}[INFO]{Colors.ENDC} {text}")


def print_success(text: str):
    """Print success message"""
    print(f"{Colors.OKGREEN}[SUCCESS]{Colors.ENDC} {text}")


def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.WARNING}[WARNING]{Colors.ENDC} {text}")


def print_error(text: str):
    """Print error message"""
    print(f"{Colors.FAIL}[ERROR]{Colors.ENDC} {text}")


def print_key_value(key: str, value: Any, color: str = Colors.OKCYAN):
    """Print a key-value pair with formatting"""
    print(f"{color}{Colors.BOLD}{key:30s}{Colors.ENDC}: {value}")


def print_metric(name: str, value: float, unit: str = "", color: str = Colors.OKGREEN):
    """Print a metric with colored formatting"""
    print(f"{color}{Colors.BOLD}{name:25s}{Colors.ENDC}: {value:.4f} {unit}")


def print_progress(epoch: int, total: int, metrics: dict):
    """Print training progress with colored formatting"""
    print(f"\n{Colors.BOLD}Epoch [{epoch}/{total}]{Colors.ENDC}")
    for key, value in metrics.items():
        if isinstance(value, float):
            print_key_value(key, f"{value:.4f}", Colors.OKCYAN)
        else:
            print_key_value(key, str(value), Colors.OKCYAN)


def suppress_warnings():
    """Suppress all warnings"""
    warnings.filterwarnings('ignore')
    # Suppress MNE specific warnings
    os.environ['MNE_LOGGING_LEVEL'] = 'ERROR'
    try:
        import mne
        mne.set_log_level('ERROR')
    except:
        pass


# Initialize: suppress warnings at module import
suppress_warnings()





