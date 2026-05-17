"""Professional visualization utilities."""
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class VisualizationHelper:
    @staticmethod
    def setup_style():
        sns.set_style("whitegrid")
        plt.rcParams["figure.figsize"] = (12, 8)
        plt.rcParams["font.size"] = 11
    
    @staticmethod
    def save_figure(fig, save_path):
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
