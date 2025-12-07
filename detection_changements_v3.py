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
    try:
        with rasterio.open(chemin) as src:
            data = src.read(1)
            profile = src.profile
            transform = src.transform
            crs = src.crs
            return data, profile, transform, crs
    except FileNotFoundError:
        print(f"✗ Fichier manquant: {chemin}")
        raise
    except rasterio.errors.RasterioIOError as e:
        print(f"✗ Erreur de lecture raster {chemin}: {e}")
        raise

def detecter_changements(indice_t1, indice_t2, seuil):
    """Détecte les changements entre deux dates en tenant compte des NoData"""
    mask_valid = (~np.isnan(indice_t1)) & (~np.isnan(indice_t2))
    delta = np.full_like(indice_t1, np.nan, dtype=np.float32)
    delta[mask_valid] = indice_t2[mask_valid] - indice_t1[mask_valid]

    changements = np.zeros_like(delta, dtype=np.uint8)
    changements[(delta < -seuil) & mask_valid] = 1  # Dégradation
    changements[(delta > seuil) & mask_valid] = 2   # Amélioration

    stats = {
        'degradation': int(np.sum(changements == 1)),
        'amelioration': int(np.sum(changements == 2)),
        'stable': int(np.sum(changements == 0)),
        'delta_moyen': float(np.nanmean(delta))
    }
    return changements, delta, stats

def sauvegarder_raster(data, profile, nom):
    """Sauvegarde un raster"""
    profile.update(dtype=rasterio.float32, count=1, compress='lzw')
    out_path = os.path.join(OUTPUT_DIR, nom)
    try:
        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(data.astype(rasterio.float32), 1)
        print(f"✓ Raster sauvegardé: {nom}")
    except Exception as e:
        print(f"✗ Erreur sauvegarde raster {nom}: {e}")
        raise

def visualiser_changements(changements, delta, titre, nom_fichier):
    """Crée et sauvegarde les visualisations"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Carte de changements
    colors = ['gray', 'red', 'green']
    from matplotlib.colors import ListedColormap
    im1 = axes[0].imshow(changements, cmap=ListedColormap(colors), vmin=0, vmax=2)
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
    out_path = os.path.join(OUTPUT_DIR, nom_fichier)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Visualisation sauvegardée: {nom_fichier}")

def vectoriser_changements(changements, transform, crs, nom_fichier, format_export='shp'):
    """
    Vectorise et exporte les zones de changements.
    
    Parameters:
    -----------
    changements : np.array
        Raster de changements (0=stable, 1=dégradation, 2=amélioration)
    transform : affine.Affine
        Transformation du raster
    crs : dict
        CRS du raster
    nom_fichier : str
        Nom du fichier à sauvegarder (sans extension)
    format_export : str
        'shp' pour Shapefile, 'geojson' pour GeoJSON
    """
    mask = changements > 0
    resultats = []

    for geom, val in shapes(changements.astype(np.int32), mask=mask, transform=transform):
        poly = shape(geom)
        if poly.area >= 400:  # seuil 400 m²
            resultats.append({
                'geometry': poly,
                'type': int(val),
                'label': 'Dégradation' if val == 1 else 'Amélioration',
                'superficie_ha': poly.area / 10000
            })

    if resultats:
        gdf = gpd.GeoDataFrame(resultats, crs=crs)
        out_path = os.path.join(OUTPUT_DIR, f"{nom_fichier}.{format_export}")
        try:
            if format_export.lower() == 'geojson':
                gdf.to_file(out_path, driver='GeoJSON')
            else:
                gdf.to_file(out_path)
            print(f"✓ {format_export.upper()} exporté: {out_path} ({len(gdf)} polygones)")
        except Exception as e:
            print(f"✗ Erreur export {format_export}: {e}")
            raise
        return gdf
    return None

# 
############################# ANALYSES
# 

def analyse_ndvi_simple(annee1, annee2):
    """Analyse NDVI simple entre deux années"""
    print(f"\n{'='*60}\nANALYSE NDVI: {annee1} → {annee2}\n{'='*60}")

    ndvi1, profile, transform, crs = charger_indice('NDVI', annee1)
    ndvi2, _, _, _ = charger_indice('NDVI', annee2)

    changements, delta, stats = detecter_changements(ndvi1, ndvi2, SEUIL_NDVI)

    print(f"Résultats NDVI {annee1}-{annee2}: Dégradation={stats['degradation']:,}, Amélioration={stats['amelioration']:,}, Stable={stats['stable']:,}, Delta moyen={stats['delta_moyen']:.4f}")

    sauvegarder_raster(delta, profile, f'delta_NDVI_{annee1}_{annee2}.tif')
    sauvegarder_raster(changements, profile, f'changements_NDVI_{annee1}_{annee2}.tif')
    visualiser_changements(changements, delta, f'NDVI {annee1}-{annee2}', f'carte_NDVI_{annee1}_{annee2}.png')
    vectoriser_changements(changements, transform, crs, f'changements_NDVI_{annee1}_{annee2}.shp')

def analyse_clustering(annee1, annee2, n_clusters=4):
    """Clustering K-means sur tous les indices"""
    print(f"\n{'='*60}\nCLUSTERING: {annee1} → {annee2}\n{'='*60}")

    indices_data = {}
    for indice in ['NDVI', 'NBR', 'NDWI']:
        ind1, profile, transform, crs = charger_indice(indice, annee1)
        ind2, _, _, _ = charger_indice(indice, annee2)
        indices_data[indice] = (ind1, ind2)

    # Calculer deltas avec NoData
    h, w = next(iter(indices_data.values()))[0].shape
    features = []
    for indice in ['NDVI', 'NBR', 'NDWI']:
        delta = indices_data[indice][1] - indices_data[indice][0]
        features.append(delta.flatten())
    features = np.stack(features, axis=1)
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
    plt.savefig(os.path.join(OUTPUT_DIR, f'clusters_{annee1}_{annee2}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Visualisation clusters sauvegardée: clusters_{annee1}_{annee2}.png")

def analyse_multi_indices(annee1, annee2):
    """Analyse combinée de tous les indices"""
    print(f"\n{'='*60}\nANALYSE MULTI-INDICES: {annee1} → {annee2}\n{'='*60}")

    indices = ['NDVI', 'NBR', 'NDWI']
    seuils = [SEUIL_NDVI, SEUIL_NBR, SEUIL_NDWI]

    for indice, seuil in zip(indices, seuils):
        try:
            ind1, profile, _, _ = charger_indice(indice, annee1)
            ind2, _, _, _ = charger_indice(indice, annee2)

            changements, delta, stats = detecter_changements(ind1, ind2, seuil)

            print(f"{indice}: Dégradation={stats['degradation']:,}, Amélioration={stats['amelioration']:,} px")

            sauvegarder_raster(delta, profile, f'delta_{indice}_{annee1}_{annee2}.tif')
        except (FileNotFoundError, rasterio.errors.RasterioIOError) as e:
            print(f"✗ Erreur avec {indice}: {e}")

# 
####################### EXÉCUTION
# 

if __name__ == "__main__":
    print("\n" + "="*60)
    print("GMQ710 - DÉTECTION DE CHANGEMENTS MONTÉRÉGIE")
    print("="*60)

    # Dictionnaire pour récapitulatif global
    recap_global = {'degradation': 0, 'amelioration': 0}

    # Boucle temporelle sur toutes les années consécutives
    for i in range(len(ANNEES)-1):
        annee1, annee2 = ANNEES[i], ANNEES[i+1]

        # NDVI simple
        ndvi1, profile, transform, crs = charger_indice('NDVI', annee1)
        ndvi2, _, _, _ = charger_indice('NDVI', annee2)
        changements, delta, stats = detecter_changements(ndvi1, ndvi2, SEUIL_NDVI)
        
        recap_global['degradation'] += stats['degradation']
        recap_global['amelioration'] += stats['amelioration']

        analyse_ndvi_simple(annee1, annee2)
        analyse_clustering(annee1, annee2, n_clusters=4)
        analyse_multi_indices(annee1, annee2)

    print("\n" + "="*60)
    print("RÉCAPITULATIF GLOBAL NDVI (toutes années consécutives):")
    print(f"  Dégradation totale: {recap_global['degradation']:,} pixels")
    print(f"  Amélioration totale: {recap_global['amelioration']:,} pixels")
    print("="*60)

    print("\nTERMINÉ!")
    print(f"Résultats dans: {OUTPUT_DIR}/")
    print("="*60 + "\n")



