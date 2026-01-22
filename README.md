# Suez2
Project for AI4Industry 2026

Projet de prédiction du niveau d'eau des nappes alluviales en collaboration avec SUEZ

Ici se trouve une description de tous les fichiers et de leurs utilisations.

Le dossier Dataset est vide, pour récupérer les données, contactez

Le dossier dataset s'organise de la manière suivante : 
- dataset
  - static_attributes
    - geology_attributes_bh.csv
    - hydrogeology_attributes_bh.csv
    - piezo_characs.csv
    - poster_sig.csv
    - soil_general_attributes_bh.csv
  - times_series
    - forçages
      - ***.txt
    - piezos
      - ***.csv

# Description des fichiers

  - DataPreprocessing.py

    Permet de fusionner et de nettoyer les donnéers d'un même site présent dans le dataset. DataPreprocessing.ipynb est un exemple d'utilisation. Ce type de fichier n'est pas implémenté dans tous les models, certains demandes de rentrer le nom des deux fichiers. Se trouvent dans le dossier src

  - lstm30j-7j.py

    Permet d'utiliser le model lstm pour prédire sur le long terme à l'aide des présentes dataset/time_series. Paramètres ajustables

  - Dual_lstm_model.ipynb

    Permet d'utiliser le model Dual lstm pour prédire sur le long terme à l'aide des présentes dans dataset/time_series. Paramètres ajustables. Plus précis que le lstm

  - one_step_transformer.ipynb

    Permet d'utiliser le model one step transformer pour prédire sur le long terme à l'aide des présentes dans dataset/time_series. Paramètres ajustables. Plus précis que le lstm et le dual lstm

  - StaticModelRegression.py

    Permet d'utiliser un model de Regression Static pour prédire les nappes alluviales d'un site en fonction de ses caractéristiques trouvé dans dataset/static_attributes.

  - Sarima.ipynb

    Permet d'utiliser le model Sarima pour prédire sur le long terme à l'aide des présentes dans dataset/time_series. 

  - Reservoirs

    Dossier contenant une utilisation du model reservoir. Utiliser pour des prédictions précises sur une courte durée.



# Contributeurs

Wassim AARAB

Omar EL ALAOUI EL ISLAMILI

Abdelhafid GUEROUANI

Florent CRAHAY--BOUDOU

Mathis PEREIRA PEDRO 

Hugo BASTIEN

Pierre-Antoine CASSARD

Thibault BUTEAU 

Hugo BILLERIT

Elric MARIANO

Lilian DELETOILE

Iskander HADJI 

Victor AUDUREAU

