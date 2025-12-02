"""
############################### GMQ710 - Détection de changements en Montérégie #############################################
"""

import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from skimage.filters import threshold_otsu

#
################# CONFIGURATION
# 

# Chemins d'accès directs
DATA_DIR = "/home/snabraham6/#projet_de_session_gmq710/data"
OUTPUT_DIR = "/home/snabraham6/#projet_de_session_gmq710/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Paramètres
SEUIL_NDVI = 0.15
SEUIL_NBR = 0.20
SEUIL_NDWI = 0.10
ANNEES = [2015, 2016, 2017, 2018, 2019, 2020]

# 
##################### FONCTIONS DE BASE
# 

def charger_indice(indice, annee):
    """Charge un indice pour une année donnée"""
    chemin = f"{DATA_DIR}/{indice}_{annee}.tif"
    with rasterio.open(chemin) as src:
        return src.read(1), src.profile, src.transform, src.crs

def detecter_changements(indice_t1, indice_t2, seuil):
    """Détecte les changements entre deux dates"""
    delta = indice_t2 - indice_t1
    changements = np.zeros_like(delta, dtype=np.uint8)
    changements[delta < -seuil] = 1  # Dégradation
    changements[delta > seuil] = 2   # Amélioration
    
    stats = {
        'degradation': int(np.sum(changements == 1)),
        'amelioration': int(np.sum(changements == 2)),
        'stable': int(np.sum(changements == 0)),
        'delta_moyen': float(np.nanmean(delta))
    }
    return changements, delta, stats

def sauvegarder_raster(data, profile, nom):
    """Sauvegarde un raster"""
    profile.update(dtype=rasterio.float32, count=1)
    with rasterio.open(f"{OUTPUT_DIR}/{nom}", 'w', **profile) as dst:
        dst.write(data.astype(rasterio.float32), 1)
    print(f"✓ {nom} sauvegardé")

def visualiser_changements(changements, delta, titre, nom_fichier):
    """Crée et sauvegarde les visualisations"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Carte de changements
    colors = ['gray', 'red', 'green']
    im1 = axes[0].imshow(changements, cmap=plt.cm.colors.ListedColormap(colors), vmin=0, vmax=2)
    axes[0].set_title(f'{titre} - Changements', fontweight='bold')
    axes[0].axis('off')
    
    from matplotlib.patches import Patch
    legend = [Patch(facecolor=colors[i], label=l) for i, l in enumerate(['Stable', 'Dégradation', 'Amélioration'])]
    axes[0].legend(handles=legend, loc='upper right')
    
    # Delta
    im2 = axes[1].imshow(delta, cmap='RdBu', vmin=-0.5, vmax=0.5)
    axes[1].set_title(f'{titre} - Delta', fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{nom_fichier}", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ {nom_fichier} sauvegardé")

def vectoriser_changements(changements, transform, crs, nom_fichier):
    """Vectorise et exporte en shapefile"""
    mask = changements > 0
    resultats = []
    
    for geom, val in shapes(changements.astype(np.int32), mask=mask, transform=transform):
        poly = shape(geom)
        if poly.area >= 100:
            resultats.append({
                'geometry': poly,
                'type': int(val),
                'label': 'Dégradation' if val == 1 else 'Amélioration',
                'superficie_ha': poly.area / 10000
            })
    
    if resultats:
        gdf = gpd.GeoDataFrame(resultats, crs=crs)
        gdf.to_file(f"{OUTPUT_DIR}/{nom_fichier}")
        print(f"✓ {nom_fichier} exporté ({len(gdf)} polygones)")
        return gdf
    return None

# 
############################# ANALYSES
# 

def analyse_ndvi_simple(annee1, annee2):
    """Analyse NDVI simple entre deux années"""
    print(f"\n{'='*60}")
    print(f"ANALYSE NDVI: {annee1} → {annee2}")
    print(f"{'='*60}")
    
    ndvi1, profile, transform, crs = charger_indice('NDVI', annee1)
    ndvi2, _, _, _ = charger_indice('NDVI', annee2)
    
    changements, delta, stats = detecter_changements(ndvi1, ndvi2, SEUIL_NDVI)
    
    print(f"\nRésultats:")
    print(f"  Dégradation: {stats['degradation']:,} pixels")
    print(f"  Amélioration: {stats['amelioration']:,} pixels")
    print(f"  Stable: {stats['stable']:,} pixels")
    print(f"  Delta moyen: {stats['delta_moyen']:.4f}")
    
    sauvegarder_raster(delta, profile, f'delta_NDVI_{annee1}_{annee2}.tif')
    sauvegarder_raster(changements, profile, f'changements_NDVI_{annee1}_{annee2}.tif')
    visualiser_changements(changements, delta, f'NDVI {annee1}-{annee2}', f'carte_NDVI_{annee1}_{annee2}.png')
    vectoriser_changements(changements, transform, crs, f'changements_NDVI_{annee1}_{annee2}.shp')

def analyse_clustering(annee1, annee2, n_clusters=4):
    """Clustering K-means sur tous les indices"""
    print(f"\n{'='*60}")
    print(f"CLUSTERING: {annee1} → {annee2}")
    print(f"{'='*60}")
    
    # Charger les données
    ndvi1, profile, transform, crs = charger_indice('NDVI', annee1)
    ndvi2, _, _, _ = charger_indice('NDVI', annee2)
    nbr1, _, _, _ = charger_indice('NBR', annee1)
    nbr2, _, _, _ = charger_indice('NBR', annee2)
    ndwi1, _, _, _ = charger_indice('NDWI', annee1)
    ndwi2, _, _, _ = charger_indice('NDWI', annee2)
    
    # Calculer deltas
    delta_ndvi = ndvi2 - ndvi1
    delta_nbr = nbr2 - nbr1
    delta_ndwi = ndwi2 - ndwi1
    
    # Préparer features
    h, w = delta_ndvi.shape
    features = np.stack([delta_ndvi.flatten(), delta_nbr.flatten(), delta_ndwi.flatten()], axis=1)
    mask = ~np.isnan(features).any(axis=1)
    features_clean = features[mask]
    
    # Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(features_clean)
    
    # Reconstruction
    clusters_map = np.full(h * w, np.nan)
    clusters_map[mask] = clusters
    clusters_map = clusters_map.reshape(h, w)
    
    print(f"✓ {n_clusters} clusters détectés")
    
    # Sauvegarde
    sauvegarder_raster(clusters_map, profile, f'clusters_{annee1}_{annee2}.tif')
    
    # Visualisation
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(clusters_map, cmap='tab10')
    ax.set_title(f'Clusters de changements {annee1}-{annee2}', fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/clusters_{annee1}_{annee2}.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ clusters_{annee1}_{annee2}.png sauvegardé")

def analyse_multi_indices(annee1, annee2):
    """Analyse combinée de tous les indices"""
    print(f"\n{'='*60}")
    print(f"ANALYSE MULTI-INDICES: {annee1} → {annee2}")
    print(f"{'='*60}")
    
    indices = ['NDVI', 'NBR', 'NDWI']
    seuils = [SEUIL_NDVI, SEUIL_NBR, SEUIL_NDWI]
    
    for indice, seuil in zip(indices, seuils):
        try:
            ind1, profile, _, _ = charger_indice(indice, annee1)
            ind2, _, _, _ = charger_indice(indice, annee2)
            
            changements, delta, stats = detecter_changements(ind1, ind2, seuil)
            
            print(f"\n{indice}:")
            print(f"  Dégradation: {stats['degradation']:,} px")
            print(f"  Amélioration: {stats['amelioration']:,} px")
            
            sauvegarder_raster(delta, profile, f'delta_{indice}_{annee1}_{annee2}.tif')
        except:
            print(f"✗ Erreur avec {indice}")

# 
####################### EXÉCUTION
# 

if __name__ == "__main__":
    print("\n" + "="*60)
    print("GMQ710 - DÉTECTION DE CHANGEMENTS MONTÉRÉGIE")
    print("="*60)
    
    # Analyse NDVI 2015-2020
    analyse_ndvi_simple(2015, 2020)
    
    # Clustering 2015-2020
    analyse_clustering(2015, 2020, n_clusters=4)
    
    # Analyse multi-indices 2015-2020
    analyse_multi_indices(2015, 2020)
    
    print("\n" + "="*60)
    print("TERMINÉ!")
    print(f"Résultats dans: {OUTPUT_DIR}/")
    print("="*60 + "\n")
