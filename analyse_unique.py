# -*- coding: utf-8 -*-
"""
GMQ710 - Analyse d'un seul indice à la fois
Utilisez: python analyse_unique.py NDVI 2015 2020
"""

import sys
import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt

# Configuration
DATA_DIR = "/home/snabraham6/#projet_de_session_gmq710/data"
OUTPUT_DIR = "/home/snabraham6/#projet_de_session_gmq710/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEUILS = {'NDVI': 0.15, 'NBR': 0.20, 'NDWI': 0.10}

def analyser_indice(indice, annee1, annee2):
    """Analyse complète d'un seul indice"""
    print(f"\n{'='*60}")
    print(f"ANALYSE {indice}: {annee1} → {annee2}")
    print(f"{'='*60}\n")
    
    # Charger les données
    print("Chargement des données...")
    chemin1 = f"{DATA_DIR}/{indice}_{annee1}.tif"
    chemin2 = f"{DATA_DIR}/{indice}_{annee2}.tif"
    
    with rasterio.open(chemin1) as src:
        data1 = src.read(1)
        profile = src.profile
        transform = src.transform
        crs = src.crs
    
    with rasterio.open(chemin2) as src:
        data2 = src.read(1)
    
    print(f"✓ Données chargées: {data1.shape}")
    
    # Calculer delta
    print("Calcul des changements...")
    delta = data2 - data1
    seuil = SEUILS[indice]
    
    changements = np.zeros_like(delta, dtype=np.uint8)
    changements[delta < -seuil] = 1  # Dégradation
    changements[delta > seuil] = 2   # Amélioration
    
    # Statistiques
    stats = {
        'degradation': int(np.sum(changements == 1)),
        'amelioration': int(np.sum(changements == 2)),
        'stable': int(np.sum(changements == 0)),
        'delta_moyen': float(np.nanmean(delta)),
        'delta_min': float(np.nanmin(delta)),
        'delta_max': float(np.nanmax(delta))
    }
    
    print(f"\nRésultats:")
    print(f"  Dégradation: {stats['degradation']:,} px ({stats['degradation']/data1.size*100:.2f}%)")
    print(f"  Amélioration: {stats['amelioration']:,} px ({stats['amelioration']/data1.size*100:.2f}%)")
    print(f"  Stable: {stats['stable']:,} px ({stats['stable']/data1.size*100:.2f}%)")
    print(f"  Delta: [{stats['delta_min']:.3f}, {stats['delta_max']:.3f}], moyenne: {stats['delta_moyen']:.3f}")
    
    # Sauvegarder rasters
    print("\nSauvegarde des rasters...")
    profile.update(dtype=rasterio.float32, count=1)
    
    with rasterio.open(f"{OUTPUT_DIR}/delta_{indice}_{annee1}_{annee2}.tif", 'w', **profile) as dst:
        dst.write(delta.astype(rasterio.float32), 1)
    print(f"✓ delta_{indice}_{annee1}_{annee2}.tif")
    
    with rasterio.open(f"{OUTPUT_DIR}/changements_{indice}_{annee1}_{annee2}.tif", 'w', **profile) as dst:
        dst.write(changements.astype(rasterio.float32), 1)
    print(f"✓ changements_{indice}_{annee1}_{annee2}.tif")
    
    # Visualisation simple
    print("\nCréation de la carte...")
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(delta, cmap='RdBu', vmin=-0.5, vmax=0.5)
    ax.set_title(f'Delta {indice} {annee1}-{annee2}', fontweight='bold', fontsize=16)
    ax.axis('off')
    plt.colorbar(im, ax=ax, label='Variation', shrink=0.8)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/carte_{indice}_{annee1}_{annee2}.png", dpi=200, bbox_inches='tight')
    plt.show()
    print(f"✓ carte_{indice}_{annee1}_{annee2}.png")
    
    print(f"\n{'='*60}")
    print("TERMINÉ!")
    print(f"{'='*60}\n")
    
    return stats

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("\nUtilisation: python analyse_unique.py INDICE ANNEE1 ANNEE2")
        print("Exemple: python analyse_unique.py NDVI 2015 2020")
        print("\nIndices disponibles: NDVI, NBR, NDWI")
        print("Années: 2015, 2016, 2017, 2018, 2019, 2020\n")
        sys.exit(1)
    
    indice = sys.argv[1].upper()
    annee1 = int(sys.argv[2])
    annee2 = int(sys.argv[3])
    
    if indice not in SEUILS:
        print(f"Erreur: {indice} invalide. Utilisez: NDVI, NBR ou NDWI")
        sys.exit(1)
    
    analyser_indice(indice, annee1, annee2)
