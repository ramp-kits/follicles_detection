# TODO

## Organisation du challenge

#### Métriques

Pour l'instant on n'a que l'AveragePrecition pour la classe "Secondary"
- [ ] il faut le calculer pour les autres classes
- [ ] les submission seront classées en fonction d'un moyenne pondérée des AP par classe

#### Data

Il faut les rendre accessibles:
- [ ] les mettre sur OSF
- [ ] créer un download_data.py et le tester
- [ ] définir une fois pour toute public vs private, train vs test


Valider le format des données qu'on fournit:
- est-ce que le fait que ça en jpg rend trop long le chargement des images ?
  alternative: tiff / np.array
- est-ce qu'on devrait resize une fois pour toute l'ensemble des images
  (finalement on utilise une résolution de 224px alors que la plupart des
   follicules sont plus grands)

#### Documentation pour les participants

- [ ] README qui explique comment installer et comment lancer une soumission
- [ ] notebook qui
    - explique les données / le challenge
    - implémente une soumission
- [ ] supprimer tous les notebooks "périmés"

## workflow RAMP

- pour l'instant on n'arrive pas à combiner les prédictions de plusieurs modèle
    - si je le fais: toutes les soumissions seront traitées par NMS même
      si le candidat n'a pas pensé à l'implémenter. Risque de confusion.
    - [ ] j'aimerais me passer de ces "bagged predictions" -> est-ce que c'est possible ?

- refactor le code qui calcule AveragePrecision qui hérite du code du notebook
  un peu bordélique
  - peut être qu'il faut re-définir le format des `y_pred` que doivent retourner
    les modèles. Aujourd'hui c'est:

    ```
    np.array(
        # first image
        [
            # list of predictions
            {"proba": 0.1928, "label": "Secondary", "bbox": (x1, y1, x2, y2)},
            {"proba": 0.1928, "label": "Secondary", "bbox": (x1, y1, x2, y2)},
        ]
        # second image
        []
    )
    ```

- [ ] est-ce qu'on peut faire mieux pour avoir un package local `ramp_custom`
  qui peut être utilisé par les participants et dans le `problem.py` ?

## Soumissions "random window classifier"

#### Entraîner le modèle

- créer un "générateur type keras" pour les vignettes d'entraînement
- entrainer le modèle plutôt que de le charger depuis un fichier
  en utilisant ce générateur (model.fit_generator())

#### Accélérer les choses

On veut permettre à un utilisateur de lancer cette soumission en
moins de 20 minutes. Les pistes sont:

- [x] configurer le `problem.py` pour utiliser "--quick-test"
  et ne charger qu'une sous-partie des données
- [x] mesurer ce qui prend du temps dans les prédictions.
      voir [cette issue](https://github.com/frcaud/follicles_detection/issues/9)
- [x] "vectoriser" l'appel au model au moment de la prédiction
  Pour chaque image:
    - on génère une liste de tuple de boxes à classifier
    - on genère un tensor d'images cropped et resized (ligne = cropped image)
        dimensions = (n_cropping_boxes x 224 x 224 x 3)
        (comment faire ça non séquentiellement ???)
    - on applique une seule fois model.predict(tensor)
        -> sort les probas par classe sous la forme (n_cropping_boxes, n_classes)

  


#### Améliorer la qualité de prédiction

- [ ] Pour l'instant seulement les boxes de ~1000px -> on ne détecte que des secondary.
Attention si on met des petites boxes ça va prendre plus longtemps.
