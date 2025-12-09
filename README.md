# GMQ710 - Détection automatisée de changements d'occupation du sol en Montérégie

## Objectifs

Développer une chaîne automatisée pour identifier et cartographier les changements d'occupation du sol en Montérégie entre 2015 et 2020 à partir d'images Sentinel-2. Le projet cible la détection de coupes forestières, feux, expansion urbaine et dégradation végétale en combinant des méthodes simples de seuillage avec des techniques avancées de classification et segmentation.

## Données utilisées

| Source | Type | Format | Utilité |
|--------|------|--------|---------|
| Copernicus Sentinel-2 (L2A) | Raster multispectral (~10 m) | GeoTIFF | Calcul des indices NDVI, NBR, NDWI et détection des variations temporelles |
| Limites administratives Montérégie | Vecteur | Shapefile | Délimitation de la zone d'étude et découpage des images |
| Carte d'occupation du sol 2020 (MELCCFP) | Raster | GeoTIFF | Validation et comparaison des résultats de détection |

Les indices sont disponibles ici : C:\Users\DINO\OneDrive\Documents\gmq710

## Approche / Méthodologie envisagée

La démarche se déroule en cinq étapes progressives allant du simple vers le complexe.

**Étape 1 - Préparation des données:** Téléchargement des images Sentinel-2 multi-temporelles via Google Earth Engine, découpage selon l'emprise de la Montérégie, reprojection en MTM Zone 8 et application de masques de nuages basés sur la couche SCL.

**Étape 2 - Calcul des indices spectraux:** Sur GEE, nous calculons le NDVI pour la vigueur végétale, le NBR pour les perturbations forestières et le NDWI pour les variations d'humidité. Les variations temporelles de ces indices forment la base de la détection.

**Étape 3 - Détection simple:** Application de seuils sur les différences ou ratios d'indices entre deux dates pour isoler les zones de changement significatif et produire une classification binaire initiale.

**Étape 4 - Méthodes avancées:** Segmentation avec scikit-image, classification supervisée avec RandomForest et clustering K-means pour révéler des changements subtils. Comparaison des approches selon leur cohérence spatiale et taux de fausses détections.

**Étape 5 - Production et exportation:** Génération de rasters de variation et vectorisation des zones de changement en shapefiles/GeoJSON. Visualisation finale dans QGIS avec statistiques par type de changement et par secteur.

## Outils et langages prévus

**Langage principal:** Python

**Bibliothèques Python utilisées:**
- numpy, pandas : calculs matriciels et statistiques
- rasterio, xarray, rioxarray : manipulation de rasters
- geopandas, shapely : analyses vectorielles et géométriques
- matplotlib : visualisation
- scikit-image, scikit-learn : segmentation, classification et clustering
- os, glob : automatisation

**Logiciel SIG:** QGIS pour validation visuelle et cartographie finale

**Plateforme cloud:** Google Earth Engine pour extraction et calcul des indices

## Répartition des tâches dans l'équipe

**Membre 1 (Ibrahima):** Extraction Sentinel-2 sur GEE, calcul des indices, prétraitements et masques de nuages

**Membre 2 (Nourredine):** Développement de la chaîne Python, méthodes de détection, analyses avancées et intégration SIG

**Travail commun:** Interprétation des résultats, validation croisée, rédaction du rapport et production cartographique

## Questions à résoudre

Quel secteur de la Montérégie prioriser pour l'analyse détaillée? Saint-Hyacinthe, Rouville et Brome-Missisquoi présentent des dynamiques territoriales distinctes.

Quels seuils privilégier pour minimiser les fausses détections tout en capturant les vrais changements?

Quelle méthode avancée offrira le meilleur compromis précision-robustesse? La classification supervisée nécessite des données d'entraînement, le clustering est non supervisé mais produit des groupes à interpréter.

Comment optimiser les temps de traitement avec plusieurs années de données à 10 mètres de résolution?

Faut-il intégrer des sources externes pour valider les détections? Permis de coupe, registres de feux ou plans d'urbanisme pourraient renforcer la fiabilité.
