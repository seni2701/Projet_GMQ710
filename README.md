# GMQ710 - Détection automatisée de changements d'occupation du sol en Montérégie

## Objectifs

Développer une chaîne automatisée pour identifier et cartographier les changements d'occupation du sol en Montérégie entre 2015 et 2020 à partir d'images Sentinel-2. Le projet cible la détection de coupes forestières, feux, expansion urbaine et dégradation végétale en combinant des méthodes simples de seuillage avec des techniques avancées de classification et segmentation.

## Données utilisées

| Source | Type | Format | Utilité |
|--------|------|--------|---------|
| Copernicus Sentinel-2 (L2A) | Raster multispectral (~10 m) | GeoTIFF | Calcul des indices NDVI, NBR, NDWI et détection des variations temporelles |
| Limites administratives Montérégie | Vecteur | Shapefile | Délimitation de la zone d'étude et découpage des images |


## Approche / Méthodologie envisagée

La démarche se déroule en cinq étapes progressives allant du simple vers le complexe.

**Étape 1 - Préparation des données:** Téléchargement des images Sentinel-2 multi-temporelles via Google Earth Engine, découpage selon l'emprise de la Montérégie, reprojection en MTM Zone 8 et application de masques de nuages basés sur la couche SCL.

**Étape 2 - Calcul des indices spectraux:** Sur GEE, nous calculons le NDVI pour la vigueur végétale, le NBR pour les perturbations forestières et le NDWI pour les variations d'humidité. Les variations temporelles de ces indices forment la base de la détection.

**Étape 3 - Détection simple:** Application de seuils empiriques sur les différences d’indices (ΔNDVI, ΔNBR, ΔNDWI) entre deux dates consécutives afin d’identifier les zones de changement significatif. Cette approche produit une classification binaire (changement / non-changement) servant de référence simple.

**Étape 4 - Méthodes avancées:** 

**Étape 4.1 - Segmentation spatiale (scikit-image):** Une segmentation basée sur la similarité spectrale et spatiale est appliquée aux cartes de variations d’indices afin de regrouper les pixels en objets homogènes. Les changements sont ensuite détectés à l’échelle des segments, ce qui permet de réduire le bruit et d’améliorer la cohérence spatiale.

**Étape 4.2 - Clustering non supervisé (K-means):** Les variations multi-indices (ΔNDVI, ΔNBR, ΔNDWI) sont combinées dans un espace multi-dimensionnel et regroupées par clustering K-means. Les clusters sont interprétés a posteriori comme différents types ou intensités de changement.

**Étape 4.3 - Comparaison des approches avancées:** Les approches de segmentation et de clustering sont comparées selon : leur cohérence spatiale, la stabilité des zones détectées d’une année à l’autre, et leur sensibilité au bruit par rapport à la détection simple par seuils.

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

**Travail commun:** Interprétation des résultats, rédaction du rapport et production cartographique

## Questions à résoudre

Quel secteur de la Montérégie prioriser pour l'analyse détaillée? Saint-Hyacinthe, Rouville et Brome-Missisquoi présentent des dynamiques territoriales distinctes.

Quels seuils privilégier pour minimiser les fausses détections tout en capturant les vrais changements?

Quelle méthode avancée offrira le meilleur compromis précision-robustesse? La classification supervisée nécessite des données d'entraînement, le clustering est non supervisé mais produit des groupes à interpréter.

Comment optimiser les temps de traitement avec plusieurs années de données à 10 mètres de résolution?

Faut-il intégrer des sources externes pour valider les détections? Permis de coupe, registres de feux ou plans d'urbanisme pourraient renforcer la fiabilité.
