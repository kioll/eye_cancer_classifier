# Analyse de la Maladie Oculaire avec des Images (Ocula)

# Partie Technique

# Execution du code sans erreurs

Pour analyser si vous avez une maladie oculaire, vous devez installer certaines bibliothèques. Vous pouvez les installer en utilisant la commande pip :
pip install flask
pip install pillow
Ajoutez toutes les bibliothèques répertoriées dans le fichier requirements.txt qui se trouve dans le dossier web_masterC-2.

Une fois avoir installer ces deux bibliothèques
Vous devez exécuter le fichier "app_flask_local.py" qui se trouve dans le répertoire web_masterC-2.

Celui ci va lancer votre application a partir de laquelle vous pourrez importer la photo de votre oeil à analyser 

# Ce que cela crée sur votre machine

Pour créer notre modèle de prédiction, vous pouvez tester également ;) : il suffit d'executerle fichier data.ipynb qui sez trouve dans le dossier archive.zip directement.
Cela va créer le my_model_ml.h5 dans le dossier web_masterC-2.

# Autres informations 

Les fichiers app_flask.py, install_app.sh et install_conda.sh vont servir à lexecution de l'application dans le cloud sur AWS, mais vous en aurez pas besoin pour l'executer, vous, en local.


# Comment Utiliser Notre Solution ?

- Tout d'abord, vous devez télécharger une image au format PNG de votre œil en utilisant le bouton "Télécharger" de l'application.
- Une fois que vous avez téléchargé l'image, cliquez sur le bouton "Analyser votre œil" pour déterminer si vous avez une maladie oculaire et laquelle.
Vous pourrez ainsi savoir si vous avez une maladie oculaire et quelle est la nature de cette maladie.


L'objectif de notre projet est de permettre aux utilisateurs de télécharger une image au format PNG de leur œil pour déterminer s'ils ont une maladie oculaire et, le cas échéant, de quel type de maladie il s'agit. Le code que vous avez fourni semble utiliser des bibliothèques liées à l'apprentissage automatique et à l'analyse d'images pour réaliser cette tâche.
