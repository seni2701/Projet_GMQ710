"""
============================================================================
PROJET GMQ710 - DÉTECTION DE CHANGEMENTS EN MONTÉRÉGIE
Script Python complet intégrant toutes les fonctionnalités
============================================================================
"""

import os
import glob
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from rasterio.features import shapes
import xarray as xr
import rioxarray as rxr
import geopandas as gpd
from shapely.geometry import shape
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from skimage.filters import threshold_otsu
from skimage.segmentation import felzenszwalb


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration centralisée du projet"""
    
    # Chemins
    DATA_DIR = "data"
    INDICES_DIR = os.path.join(DATA_DIR, "indices")
    SHAPEFILE_DIR = os.path.join(DATA_DIR, "shapefiles")
    OUTPUT_DIR = "outputs"
    
    # Paramètres de détection
    SEUIL_NDVI = 0.15
    SEUIL_NBR = 0.20
    SEUIL_NDWI = 0.10
    
    # Années d'analyse
    ANNEES = [2015, 2016, 2017, 2018, 2019, 2020]
    
    # Projection
    CRS_MTM8 = "EPSG:32188"
    
    # Visualisation
    COLORMAP = "RdYlGn"
    DPI_EXPORT = 300
    
    @classmethod
    def creer_dossiers(cls):
        """Crée les dossiers nécessaires"""
        for directory in [cls.DATA_DIR, cls.INDICES_DIR, cls.SHAPEFILE_DIR, cls.OUTPUT_DIR]:
            os.makedirs(directory, exist_ok=True)


# ============================================================================
# CHARGEMENT DES DONNÉES
# ============================================================================

class DataLoader:
    """Gestion du chargement des données raster"""
    
    @staticmethod
    def charger_raster(chemin_fichier):
        """Charge un fichier raster GeoTIFF"""
        with rasterio.open(chemin_fichier) as src:
            data = src.read(1)
            profile = src.profile
            transform = src.transform
            crs = src.crs
        return data, profile, transform, crs
    
    @staticmethod
    def charger_indices_annuels(annee):
        """Charge les indices NDVI, NBR, NDWI pour une année"""
        indices = {}
        fichiers = {
            'ndvi': f'{Config.INDICES_DIR}/NDVI_{annee}.tif',
            'nbr': f'{Config.INDICES_DIR}/NBR_{annee}.tif',
            'ndwi': f'{Config.INDICES_DIR}/NDWI_{annee}.tif'
        }
        
        for nom, chemin in fichiers.items():
            if os.path.exists(chemin):
                indices[nom], _, _, _ = DataLoader.charger_raster(chemin)
                print(f"✓ {nom.upper()} {annee} chargé")
            else:
                print(f"✗ {chemin} introuvable")
        
        return indices
    
    @staticmethod
    def lister_fichiers_disponibles():
        """Liste tous les fichiers disponibles"""
        fichiers = glob.glob(f'{Config.INDICES_DIR}/*.tif')
        print(f"\n{'='*60}")
        print(f"Fichiers disponibles: {len(fichiers)}")
        print(f"{'='*60}")
        for f in sorted(fichiers):
            print(f"  → {os.path.basename(f)}")
        return fichiers
    
    @staticmethod
    def decouper_par_shapefile(chemin_raster, chemin_shapefile):
        """Découpe un raster selon un shapefile"""
        gdf = gpd.read_file(chemin_shapefile)
        
        with rasterio.open(chemin_raster) as src:
            geometries = [feature["geometry"] for feature in gdf.__geo_interface__["features"]]
            out_image, out_transform = mask(src, geometries, crop=True)
            out_meta = src.meta.copy()
            out_meta.update({
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })
        
        return out_image[0], out_meta


# ============================================================================
# DÉTECTION DE CHANGEMENTS SIMPLE
# ============================================================================

class ChangeDetector:
    """Méthodes de détection de changements simples"""
    
    @staticmethod
    def calculer_delta(indice_t1, indice_t2):
        """Calcule la différence entre deux dates"""
        return indice_t2 - indice_t1
    
    @staticmethod
    def calculer_ratio(indice_t1, indice_t2, epsilon=1e-6):
        """Calcule le ratio entre deux dates"""
        return (indice_t2 + epsilon) / (indice_t1 + epsilon)
    
    @staticmethod
    def detecter_changements_seuil(delta, seuil):
        """
        Détecte les changements selon un seuil
        0 = stable, 1 = dégradation, 2 = amélioration
        """
        changements = np.zeros_like(delta, dtype=np.uint8)
        changements[delta < -seuil] = 1  # Dégradation
        changements[delta > seuil] = 2   # Amélioration
        return changements
    
    @staticmethod
    def calculer_statistiques(changements, delta):
        """Calcule les statistiques de changement"""
        stats = {
            'degradation_pixels': int(np.sum(changements == 1)),
            'amelioration_pixels': int(np.sum(changements == 2)),
            'stable_pixels': int(np.sum(changements == 0)),
            'delta_mean': float(np.nanmean(delta)),
            'delta_std': float(np.nanstd(delta)),
            'delta_min': float(np.nanmin(delta)),
            'delta_max': float(np.nanmax(delta))
        }
        return stats
    
    @staticmethod
    def detecter_changements_ndvi(ndvi_t1, ndvi_t2, seuil=None):
        """Pipeline complet de détection NDVI"""
        if seuil is None:
            seuil = Config.SEUIL_NDVI
        
        delta_ndvi = ChangeDetector.calculer_delta(ndvi_t1, ndvi_t2)
        changements = ChangeDetector.detecter_changements_seuil(delta_ndvi, seuil)
        stats = ChangeDetector.calculer_statistiques(changements, delta_ndvi)
        
        return changements, delta_ndvi, stats
    
    @staticmethod
    def sauvegarder_raster(data, profile, nom_fichier):
        """Sauvegarde un raster en GeoTIFF"""
        profile_copy = profile.copy()
        profile_copy.update(dtype=rasterio.float32, count=1)
        
        chemin = f"{Config.OUTPUT_DIR}/{nom_fichier}"
        with rasterio.open(chemin, 'w', **profile_copy) as dst:
            dst.write(data.astype(rasterio.float32), 1)
        
        print(f"✓ Sauvegardé: {chemin}")


# ============================================================================
# MÉTHODES AVANCÉES
# ============================================================================

class AdvancedMethods:
    """Méthodes avancées: segmentation, classification, clustering"""
    
    @staticmethod
    def segmentation_felzenszwalb(delta_ndvi, scale=100, sigma=0.5, min_size=50):
        """Segmentation d'image"""
        data_norm = (delta_ndvi - np.nanmin(delta_ndvi)) / (np.nanmax(delta_ndvi) - np.nanmin(delta_ndvi))
        data_3d = np.stack([data_norm] * 3, axis=-1)
        segments = felzenszwalb(data_3d, scale=scale, sigma=sigma, min_size=min_size)
        return segments
    
    @staticmethod
    def seuillage_otsu(delta_ndvi):
        """Seuillage automatique par méthode Otsu"""
        data_valide = delta_ndvi[~np.isnan(delta_ndvi)]
        
        if len(data_valide) == 0:
            return np.zeros_like(delta_ndvi), 0
        
        seuil = threshold_otsu(data_valide)
        binaire = (delta_ndvi > seuil).astype(np.uint8)
        return binaire, seuil
    
    @staticmethod
    def preparer_features(delta_ndvi, delta_nbr, delta_ndwi, labels=None):
        """Prépare les features pour classification/clustering"""
        h, w = delta_ndvi.shape
        
        features = np.stack([
            delta_ndvi.flatten(),
            delta_nbr.flatten(),
            delta_ndwi.flatten()
        ], axis=1)
        
        mask_valide = ~np.isnan(features).any(axis=1)
        features_clean = features[mask_valide]
        
        if labels is not None:
            labels_clean = labels.flatten()[mask_valide]
            return features_clean, labels_clean, mask_valide, (h, w)
        
        return features_clean, mask_valide, (h, w)
    
    @staticmethod
    def classification_random_forest(features_train, labels_train, features_test=None, n_estimators=100):
        """Classification supervisée avec Random Forest"""
        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
        clf.fit(features_train, labels_train)
        
        if features_test is not None:
            predictions = clf.predict(features_test)
            return clf, predictions
        
        return clf
    
    @staticmethod
    def clustering_kmeans(features, n_clusters=3):
        """Clustering K-means"""
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(features)
        return clusters, kmeans
    
    @staticmethod
    def reconstruire_carte(predictions, mask, shape):
        """Reconstruit une carte spatiale depuis les prédictions"""
        pred_map = np.full(shape[0] * shape[1], np.nan)
        pred_map[mask] = predictions
        pred_map = pred_map.reshape(shape)
        return pred_map


# ============================================================================
# VISUALISATION
# ============================================================================

class Visualizer:
    """Génération de visualisations"""
    
    @staticmethod
    def visualiser_delta(delta, titre="Delta Indice", vmin=-0.5, vmax=0.5):
        """Visualise la différence d'un indice"""
        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(delta, cmap=Config.COLORMAP, vmin=vmin, vmax=vmax)
        ax.set_title(titre, fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Variation')
        ax.axis('off')
        plt.tight_layout()
        return fig
    
    @staticmethod
    def visualiser_changements(changements, titre="Détection de changements"):
        """Visualise la carte de changements"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = ['gray', 'red', 'green']
        labels = ['Stable', 'Dégradation', 'Amélioration']
        
        im = ax.imshow(changements, cmap=plt.cm.colors.ListedColormap(colors), vmin=0, vmax=2)
        ax.set_title(titre, fontsize=14, fontweight='bold')
        ax.axis('off')
        
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=colors[i], label=labels[i]) for i in range(3)]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def visualiser_comparaison(indice_t1, indice_t2, delta, titres=None):
        """Compare deux dates et leur différence"""
        if titres is None:
            titres = ['Date 1', 'Date 2', 'Différence']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        im1 = axes[0].imshow(indice_t1, cmap='RdYlGn', vmin=-1, vmax=1)
        axes[0].set_title(titres[0], fontsize=12, fontweight='bold')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], fraction=0.046)
        
        im2 = axes[1].imshow(indice_t2, cmap='RdYlGn', vmin=-1, vmax=1)
        axes[1].set_title(titres[1], fontsize=12, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], fraction=0.046)
        
        im3 = axes[2].imshow(delta, cmap='RdBu', vmin=-0.5, vmax=0.5)
        axes[2].set_title(titres[2], fontsize=12, fontweight='bold')
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2], fraction=0.046)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def visualiser_histogramme(delta, titre="Distribution des changements"):
        """Histogramme de distribution"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        data_valide = delta[~np.isnan(delta)]
        ax.hist(data_valide, bins=100, edgecolor='black', alpha=0.7, color='steelblue')
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Pas de changement')
        ax.set_xlabel('Variation', fontsize=12)
        ax.set_ylabel('Fréquence', fontsize=12)
        ax.set_title(titre, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def sauvegarder_figure(fig, nom_fichier):
        """Sauvegarde une figure"""
        chemin = f"{Config.OUTPUT_DIR}/{nom_fichier}"
        fig.savefig(chemin, dpi=Config.DPI_EXPORT, bbox_inches='tight')
        print(f"✓ Figure sauvegardée: {chemin}")
        plt.close(fig)


# ============================================================================
# EXPORT VERS SIG
# ============================================================================

class SIGExporter:
    """Exportation vers formats SIG"""
    
    @staticmethod
    def vectoriser_changements(raster_changements, transform, crs, seuil_superficie=100):
        """Convertit les zones de changement en polygones"""
        mask_change = raster_changements > 0
        
        resultats = []
        for geom, valeur in shapes(raster_changements.astype(np.int32), mask=mask_change, transform=transform):
            poly = shape(geom)
            superficie = poly.area
            
            if superficie >= seuil_superficie:
                resultats.append({
                    'geometry': poly,
                    'type_changement': int(valeur),
                    'superficie_m2': superficie,
                    'superficie_ha': superficie / 10000
                })
        
        if len(resultats) > 0:
            gdf = gpd.GeoDataFrame(resultats, crs=crs)
            type_labels = {1: 'Dégradation', 2: 'Amélioration'}
            gdf['label'] = gdf['type_changement'].map(type_labels)
            return gdf
        
        return None
    
    @staticmethod
    def exporter_shapefile(gdf, nom_fichier):
        """Exporte en shapefile"""
        chemin = f"{Config.OUTPUT_DIR}/{nom_fichier}"
        gdf.to_file(chemin)
        print(f"✓ Shapefile: {chemin}")
    
    @staticmethod
    def exporter_geojson(gdf, nom_fichier):
        """Exporte en GeoJSON"""
        chemin = f"{Config.OUTPUT_DIR}/{nom_fichier}"
        gdf.to_file(chemin, driver='GeoJSON')
        print(f"✓ GeoJSON: {chemin}")
    
    @staticmethod
    def generer_statistiques(gdf):
        """Génère des statistiques"""
        stats = {
            'nombre_polygones': len(gdf),
            'superficie_totale_ha': gdf['superficie_ha'].sum(),
            'superficie_moyenne_ha': gdf['superficie_ha'].mean(),
            'superficie_max_ha': gdf['superficie_ha'].max()
        }
        
        if 'label' in gdf.columns:
            stats['par_type'] = gdf.groupby('label')['superficie_ha'].agg(['count', 'sum']).to_dict()
        
        return stats


# ============================================================================
# PIPELINE PRINCIPAL
# ============================================================================

class Pipeline:
    """Pipeline complet d'analyse"""
    
    @staticmethod
    def analyse_simple(annee1, annee2):
        """Analyse simple entre deux années"""
        print(f"\n{'='*60}")
        print(f"ANALYSE SIMPLE: {annee1} → {annee2}")
        print(f"{'='*60}\n")
        
        # Chargement
        indices_t1 = DataLoader.charger_indices_annuels(annee1)
        indices_t2 = DataLoader.charger_indices_annuels(annee2)
        
        if 'ndvi' not in indices_t1 or 'ndvi' not in indices_t2:
            print("✗ Erreur: fichiers NDVI manquants")
            return None
        
        # Détection
        changements, delta_ndvi, stats = ChangeDetector.detecter_changements_ndvi(
            indices_t1['ndvi'], 
            indices_t2['ndvi']
        )
        
        # Affichage stats
        print(f"\nStatistiques de changement:")
        print(f"  Pixels dégradés: {stats['degradation_pixels']:,}")
        print(f"  Pixels améliorés: {stats['amelioration_pixels']:,}")
        print(f"  Pixels stables: {stats['stable_pixels']:,}")
        print(f"  Delta moyen: {stats['delta_mean']:.4f}")
        print(f"  Delta écart-type: {stats['delta_std']:.4f}")
        
        # Visualisations
        fig1 = Visualizer.visualiser_comparaison(
            indices_t1['ndvi'], 
            indices_t2['ndvi'], 
            delta_ndvi, 
            [f'NDVI {annee1}', f'NDVI {annee2}', f'Delta NDVI']
        )
        Visualizer.sauvegarder_figure(fig1, f'comparaison_{annee1}_{annee2}.png')
        
        fig2 = Visualizer.visualiser_changements(changements, f'Changements {annee1}-{annee2}')
        Visualizer.sauvegarder_figure(fig2, f'changements_{annee1}_{annee2}.png')
        
        fig3 = Visualizer.visualiser_histogramme(delta_ndvi, f'Distribution Delta NDVI {annee1}-{annee2}')
        Visualizer.sauvegarder_figure(fig3, f'histogramme_{annee1}_{annee2}.png')
        
        return changements, delta_ndvi, stats
    
    @staticmethod
    def analyse_clustering(annee1, annee2, n_clusters=4):
        """Analyse par clustering"""
        print(f"\n{'='*60}")
        print(f"CLUSTERING K-MEANS: {annee1} → {annee2}")
        print(f"{'='*60}\n")
        
        # Chargement
        indices_t1 = DataLoader.charger_indices_annuels(annee1)
        indices_t2 = DataLoader.charger_indices_annuels(annee2)
        
        # Vérification
        required = ['ndvi', 'nbr', 'ndwi']
        if not all(k in indices_t1 and k in indices_t2 for k in required):
            print("✗ Erreur: indices manquants pour le clustering")
            return None
        
        # Calcul des deltas
        delta_ndvi = indices_t2['ndvi'] - indices_t1['ndvi']
        delta_nbr = indices_t2['nbr'] - indices_t1['nbr']
        delta_ndwi = indices_t2['ndwi'] - indices_t1['ndwi']
        
        # Préparation et clustering
        features, mask, shape = AdvancedMethods.preparer_features(delta_ndvi, delta_nbr, delta_ndwi)
        clusters, kmeans = AdvancedMethods.clustering_kmeans(features, n_clusters=n_clusters)
        
        # Reconstruction
        clusters_map = AdvancedMethods.reconstruire_carte(clusters, mask, shape)
        
        print(f"✓ Clustering effectué: {n_clusters} groupes détectés")
        
        # Visualisation
        fig = Visualizer.visualiser_changements(
            clusters_map.astype(np.uint8), 
            f'Clusters de changements {annee1}-{annee2}'
        )
        Visualizer.sauvegarder_figure(fig, f'clustering_{annee1}_{annee2}.png')
        
        return clusters_map, kmeans
    
    @staticmethod
    def analyse_complete():
        """Pipeline complet d'analyse"""
        print("\n" + "="*60)
        print("PROJET GMQ710 - DÉTECTION DE CHANGEMENTS MONTÉRÉGIE")
        print("="*60)
        
        # Créer les dossiers
        Config.creer_dossiers()
        
        # Vérifier les données
        fichiers = DataLoader.lister_fichiers_disponibles()
        
        if len(fichiers) == 0:
            print("\n✗ Aucun fichier trouvé!")
            print("\nPlacez vos fichiers GeoTIFF dans: data/indices/")
            print("Format: NDVI_YYYY.tif, NBR_YYYY.tif, NDWI_YYYY.tif")
            return
        
        # Analyses
        try:
            # Analyse simple 2015-2020
            Pipeline.analyse_simple(2015, 2020)
        except Exception as e:
            print(f"\n✗ Erreur analyse simple: {e}")
        
        try:
            # Clustering 2015-2020
            Pipeline.analyse_clustering(2015, 2020, n_clusters=4)
        except Exception as e:
            print(f"\n✗ Erreur clustering: {e}")
        
        print("\n" + "="*60)
        print("ANALYSE TERMINÉE")
        print("="*60)
        print(f"\nRésultats disponibles dans: {Config.OUTPUT_DIR}/")


# ============================================================================
# POINT D'ENTRÉE
# ============================================================================

if __name__ == "__main__":
    # Exécuter le pipeline complet
    Pipeline.analyse_complete()
    
    # Ou exécuter des analyses spécifiques:
    # Pipeline.analyse_simple(2015, 2020)
    # Pipeline.analyse_clustering(2015, 2020, n_clusters=5)
