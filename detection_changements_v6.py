############################### GMQ710 - Détection de changements en Montérégie #############################################

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

# Chemins d'accès
DATA_DIR = r"C:\Users\DINO\OneDrive\Documents\gmq710"
OUTPUT_DIR = os.path.join(DATA_DIR, "outputs")
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
    chemin = os.path.join(DATA_DIR, f"{indice}_{annee}.tif")
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
    with rasterio.open(os.path.join(OUTPUT_DIR, nom), 'w', **profile) as dst:
        dst.write(data.astype(rasterio.float32), 1)
    print(f"✓ {nom} sauvegardé")

def visualiser_changements(changements, delta, titre, nom_fichier, decim=20):
    """
    Visualisation avec décimation pour gros rasters.
    """
    import matplotlib.colors as mcolors
    from matplotlib.patches import Patch

    # Décimation
    changements_small = changements[::decim, ::decim]
    delta_small = delta[::decim, ::decim]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Carte de changements
    colors = ['gray', 'red', 'green']
    cmap = mcolors.ListedColormap(colors)
    im1 = axes[0].imshow(changements_small, cmap=cmap, vmin=0, vmax=2)
    axes[0].set_title(f'{titre} - Changements', fontweight='bold')
    axes[0].axis('off')
    legend = [Patch(facecolor=colors[i], label=l) for i, l in enumerate(['Stable', 'Dégradation', 'Amélioration'])]
    axes[0].legend(handles=legend, loc='upper right')

    # Delta
    im2 = axes[1].imshow(delta_small, cmap='RdBu', vmin=-0.5, vmax=0.5)
    axes[1].set_title(f'{titre} - Delta', fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, nom_fichier), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✓ {nom_fichier} sauvegardé (décimation x{decim})")

def vectoriser_changements(changements, transform, crs, nom_fichier, format_export='shp'):
    """
    Vectorise et exporte les zones de changements.
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
        if format_export.lower() == 'geojson':
            gdf.to_file(out_path, driver='GeoJSON')
        else:
            gdf.to_file(out_path)
        print(f"✓ {format_export.upper()} exporté: {out_path} ({len(gdf)} polygones)")
        return gdf
    return None

#
####################### ANALYSES EXISTANTES
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
    vectoriser_changements(changements, transform, crs, f'changements_NDVI_{annee1}_{annee2}', format_export='geojson')

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
    im = ax.imshow(clusters_map[::20, ::20], cmap='tab10')  # décimation pour visualisation
    ax.set_title(f'Clusters de changements {annee1}-{annee2}', fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'clusters_{annee1}_{annee2}.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✓ clusters_{annee1}_{annee2}.png sauvegardé (décimation x20)")

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
        except Exception as e:
            print(f"✗ Erreur avec {indice}: {e}")

#
####################### ANALYSES AVANCÉES
#

def analyse_avancee(annee1, annee2, n_clusters=4):
    """
    Analyse avancée pour détecter les changements subtils avec:
    - Segmentation (scikit-image)
    - Classification supervisée (RandomForest)
    - Clustering K-means
    """
    print(f"\n{'='*60}\nANALYSE AVANCÉE: {annee1} → {annee2}\n{'='*60}")

    # Charger les indices
    indices = ['NDVI', 'NBR', 'NDWI']
    data = {}
    for indice in indices:
        ind1, profile, transform, crs = charger_indice(indice, annee1)
        ind2, _, _, _ = charger_indice(indice, annee2)
        delta = ind2 - ind1
        delta[np.isnan(delta)] = 0
        data[indice] = delta

    h, w = next(iter(data.values())).shape

    # --- 1) Segmentation ---
    threshold_seg = threshold_otsu(data['NDVI'])
    seg_map = np.zeros_like(data['NDVI'], dtype=np.uint8)
    seg_map[data['NDVI'] > threshold_seg] = 2
    seg_map[data['NDVI'] < -threshold_seg] = 1
    vectoriser_changements(seg_map, transform, crs, f"segmentation_{annee1}_{annee2}", format_export='geojson')
    print(f"✓ Segmentation effectuée avec seuil Otsu={threshold_seg:.3f}")

    # --- 2) RandomForest supervisé ---
    features = np.stack([data[indice].flatten() for indice in indices], axis=1)
    mask_valid = ~np.isnan(features).any(axis=1)
    features_clean = features[mask_valid]

    labels = np.zeros(h * w, dtype=np.uint8)
    labels[data['NDVI'].flatten() > SEUIL_NDVI] = 2
    labels[data['NDVI'].flatten() < -SEUIL_NDVI] = 1
    labels_clean = labels[mask_valid]

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(features_clean, labels_clean)
    pred = np.full(h * w, 0, dtype=np.uint8)
    pred[mask_valid] = clf.predict(features_clean)
    rf_map = pred.reshape(h, w)

    vectoriser_changements(rf_map, transform, crs, f"randomforest_{annee1}_{annee2}", format_export='geojson')
    print(f"✓ RandomForest supervisé effectué")

    # --- 3) Clustering K-means ---
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = np.full(h * w, np.nan)
    clusters[mask_valid] = kmeans.fit_predict(features_clean)
    clusters_map = clusters.reshape(h, w)
    sauvegarder_raster(clusters_map, profile, f'clusters_{annee1}_{annee2}.tif')
    print(f"✓ Clustering K-means effectué ({n_clusters} clusters)")

    # --- 4) Comparaison simple ---
    total_pixels = h * w
    nb_seg = np.sum(seg_map > 0)
    nb_rf = np.sum(rf_map > 0)
    nb_km = np.sum(~np.isnan(clusters_map))

    print(f"\nComparaison méthodes (pixels détectés) :")
    print(f"  Segmentation : {nb_seg} / {total_pixels} pixels ({nb_seg/total_pixels:.1%})")
    print(f"  RandomForest : {nb_rf} / {total_pixels} pixels ({nb_rf/total_pixels:.1%})")
    print(f"  K-means      : {nb_km} / {total_pixels} pixels ({nb_km/total_pixels:.1%})")
    print("="*60)

#
####################### EXÉCUTION GLOBALE
#

if __name__ == "__main__":
    print("\n" + "="*60)
    print("GMQ710 - DÉTECTION DE CHANGEMENTS MONTÉRÉGIE")
    print("="*60)

    # Boucle sur toutes les années consécutives
    for i in range(len(ANNEES)-1):
        annee1, annee2 = ANNEES[i], ANNEES[i+1]
        try :
            # NDVI simple
            analyse_ndvi_simple(annee1, annee2)

            # Clustering
            analyse_clustering(annee1, annee2, n_clusters=4)

            # Multi-indices
            analyse_multi_indices(annee1, annee2)

        except Exception as e:
            print(f"✗ Erreur pour {annee1} → {annee2}: {e}")


    # Export récapitulatif global
    recap_df = pd.DataFrame(recap_global)
    csv_path = os.path.join(OUTPUT_DIR, "recap_global_changements.csv")
    recap_df.to_csv(csv_path, index=False)
    print(f"\n✓ Récapitulatif global sauvegardé: {csv_path}")

    print("\nTERMINÉ!")
    print(f"Résultats dans: {OUTPUT_DIR}/")
    print("="*60 + "\n")

