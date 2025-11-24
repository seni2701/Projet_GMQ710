
////////////////// Extraction indices Sentinel-2 pour Montérégie (2015-2020) ////////////////////////////////
//

////////// Définir la zone d'étude - Montérégie /////////////////////////////

var monteregie = ee.FeatureCollection("projects/steady-atlas-433820-v0/assets/100368")
  .geometry(); // Remplacer par votre asset

// Alternative: définir manuellement un rectangle approximatif
// var monteregie = ee.Geometry.Rectangle([-73.8, 45.0, -72.5, 45.8]);

///////////// Paramètres temporels /////////////////////////////////////////
var annees = [2015, 2016, 2017, 2018, 2019, 2020];
var debut_saison = '2015-06-01';  // 1er juin
var fin_saison = '2020-09-30';    // 30 septembre

//////////// Fonction de masquage des nuages pour Sentinel-2 L2A ///////////////////////////////////
function masquerNuages(image) {
  var scl = image.select('SCL');
  var masque = scl.neq(3)  // Ombres de nuages
                  .and(scl.neq(8))  // Nuages moyenne probabilité
                  .and(scl.neq(9))  // Nuages haute probabilité
                  .and(scl.neq(10)); // Cirrus
  return image.updateMask(masque);
}

//////////////// Fonction de calcul des indices spectraux //////////////////////////////////////
function calculerIndices(image) {
  // NDVI = (NIR - Red) / (NIR + Red)
  var ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI');
  
  // NBR = (NIR - SWIR2) / (NIR + SWIR2)
  var nbr = image.normalizedDifference(['B8', 'B12']).rename('NBR');
  
  // NDWI = (Green - NIR) / (Green + NIR)
  var ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI');
  
  return image.addBands([ndvi, nbr, ndwi]);
}

////////////////// Fonction principale de traitement pour une année ////////////////////////////////
function traiterAnnee(annee) {
  var debut = ee.Date.fromYMD(annee, 6, 1);
  var fin = ee.Date.fromYMD(annee, 9, 30);
  
  // Charger la collection Sentinel-2 L2A
  var collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(monteregie)
    .filterDate(debut, fin)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    .map(masquerNuages)
    .map(calculerIndices);
  
  // Créer une composite médiane
  var composite = collection.median().clip(monteregie);
  
  return composite;
}

/////////////////////////// Traiter toutes les années et exporter ///////////////////////////
annees.forEach(function(annee) {
  var composite = traiterAnnee(annee);
  
  // Exporter NDVI
  Export.image.toDrive({
    image: composite.select('NDVI'),
    description: 'NDVI_' + annee,
    folder: 'GMQ710_Monteregie',
    fileNamePrefix: 'NDVI_' + annee,
    region: monteregie,
    scale: 10,
    crs: 'EPSG:32188',  // MTM Zone 8
    maxPixels: 1e13
  });
  
  // Exporter NBR
  Export.image.toDrive({
    image: composite.select('NBR'),
    description: 'NBR_' + annee,
    folder: 'GMQ710_Monteregie',
    fileNamePrefix: 'NBR_' + annee,
    region: monteregie,
    scale: 10,
    crs: 'EPSG:32188',
    maxPixels: 1e13
  });
  
  // Exporter NDWI
  Export.image.toDrive({
    image: composite.select('NDWI'),
    description: 'NDWI_' + annee,
    folder: 'GMQ710_Monteregie',
    fileNamePrefix: 'NDWI_' + annee,
    region: monteregie,
    scale: 10,
    crs: 'EPSG:32188',
    maxPixels: 1e13
  });
  
  print('Exports configurés pour ' + annee);
});

////////////////////////////// Visualisation //////////////////////////////////////////
var composite2020 = traiterAnnee(2020);

Map.centerObject(monteregie, 9);
Map.addLayer(composite2020.select('NDVI'), 
  {min: -0.2, max: 0.8, palette: ['red', 'yellow', 'green']}, 
  'NDVI 2020');
Map.addLayer(composite2020.select('NBR'), 
  {min: -0.5, max: 0.5, palette: ['brown', 'yellow', 'darkgreen']}, 
  'NBR 2020');
Map.addLayer(composite2020.select('NDWI'), 
  {min: -0.5, max: 0.5, palette: ['brown', 'white', 'blue']}, 
  'NDWI 2020');

print('Script prêt. Cliquez sur Tasks pour lancer les exports.');
