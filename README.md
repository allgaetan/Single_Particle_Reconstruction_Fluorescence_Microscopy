# Stage M2 : Reconstruction à partir de particules isolées en microscopie par fluorescence

Ce repo reprend le [repo](https://github.com/thibaut1998e/Single_Particle_Reconstruction_Fluorescence_Microsopy) de Thibaut Eloy pour sa thèse [Reconstruction à partir de particule isolée en
microscopie par fluorescence](https://publication-theses.unistra.fr/public/theses_doctorat/2024/ELOY_Thibaut_2024_ED269.pdf). Son README original est accessible [ici](/README_FLUOFIRE.md).

Mes contributions pendant ce stage sont regroupées dans le dossier [stage_gaetan](/stage_gaetan/) et dans le fichier [end_to_end_architecture_volume.py](/reconstruction_with_dl/end_to_end_architecture_volume.py).

## Sommaire
- [Stage M2 : Reconstruction à partir de particules isolées en microscopie par fluorescence](#stage-m2--reconstruction-à-partir-de-particules-isolées-en-microscopie-par-fluorescence)
  - [Sommaire](#sommaire)
  - [Setup](#setup)
  - [Lancer une reconstruction](#lancer-une-reconstruction)
  - [Résultats](#résultats)
  - [Architecture](#architecture)

## Setup

1. Cloner le repo [Single_Particle_Reconstruction_Fluorescence_Microscopy
](https://github.com/allgaetan/Single_Particle_Reconstruction_Fluorescence_Microscopy) (dans un dossier **user/Documents/**, sinon voir l'étape 3.2)

2. Créer l'environnement virtuel avec les dépendances en exécutant la commande suivante :
```
conda env create -f single_particle_reconstruction.yml
```

3. 1. Modifier le chemin **ROOT** dans [paths.py](/stage_gaetan/paths.py) avec le home de votre user 
    2. Si le repo n'a pas été cloné dans le dossier **user/Documents/**, modifier le chemin **PATH_DOCUMENTS_FOLDER** dans [paths.py](/stage_gaetan/paths.py)  avec le dossier dans lequel a été cloné le repo

## Lancer une reconstruction

Il faut se placer à la racine du [projet](/) et lancer le script [run.py](/stage_gaetan/run.py). Ce script lit ligne par ligne les configurations décrites dans [configs.csv](/stage_gaetan/configs.csv) et les lance parallèlement sur le cluster de calcul HPC via une gestion de jobs par Slurm.

Les configurations s'écrivent au format CSV et les paramètres actuellement présents sont les suivants :

- **gen** : True/False (default True), active la génération des données synthétiques
- **rec** : True/False (default True), active la reconstruction à partir des données générées
- **prof** : True/False (default False), active le profilage du code
- **gt_name** : chaîne de caractères correspondant au nom de l'objet vérité terrain étudié (nom des fichiers dans [ground_truths/](/ground_truths/) sans l'extension .tif)
- **fourier** : True/False (default False), active la reconstruction dans le domaine de Fourier (pas encore fonctionnelle)
- **config_id** : chaîne de caractères, id de la reconstruction (si aucune, une valeur est générée avec avec le format YYYYMMDD_HHMMSS_i)
- **pth_views** : chaîne de caractères, chemin vers les vues générées (par défaut lorsque la génération est activée, le paramètre est automatiquement rempli)
- **sig_z** : float (default 5), valeur du flou anisotropique des données générées
- **nb_views** : int (default 250), nombre de vues générées
- **snr** : float (default 1), ratio signal à bruit des données générées
- **save_fold** : chaîne de caractères, chemin où est enregistré la reconstruction (par défaut, la reconstruction est enregistrée avec le dossier des vues générées dans le dossier nommé avec la **config_id**)
- **nb_epochs** : int (default 3000), nombre d'epochs de la reconstruction
- **x** : int (default 100), nombre d'epochs entre chaque checkpoint
- **timeout** : chaîne de caractères au format "HH:MM:SS" (default "1:00:00"), durée limite du job lancé sur le cluster HPC
- **nb_hps** : int (default 50), nombre d'epochs de HPS avant les **nb_epochs** - **nb_hps** epochs de SGD (lors du HPS, les checkpoints sont fait à chaque epoch indépendamment de **x**)
- **nb_kept** : int (default 8), nombre de candidats gardés à chaque itération de HPS
- **nb_children** : int (default 8), nombre de candidats générés pour chacun des **nb_kept** candidats à chaque itération de HPS (**nb_kept** x **nb_children** candidats à évalués à chaque itération)
- **nb_dir** : int (default 72), nombre de plan de rotations générés lors de la première itération de HPS
- **nb_in_plane** : int (default 12), nombre de rotation candidates générées dans chaque plan de rotation lors de la première itération de HPS (**nb_dir** x **nb_in_plane** candidats à évaluer initialement)
- **scale_factors** : chaîne de caractères au format "[float,float,float,...]" (default None), facteurs d'échelle utilisés lorsque le sous-échantillonnage progressif est utilisé. La longueur de la liste doit être égale au nombre d'itérations de HPS (par défaut 5) + 1 pour l'itération initiale. Chaque float correspond à un facteur de downsampling utilisé pour l'itération correspondante. Si un facteur est utilisé deux fois d'affilé, il est plus optimale de l'écrire une fois dans la liste puis de remplacer les suivants par None (par exemple "[0.125,0.25,0.5,1.0,None,None]" au lieu de "[0.125,0.25,0.5,1.0,1.0,1.0]"). Exemples de **scale_factors** : 
    - "[0.125,0.25,0.5,1.0,None,None]"
    - "[0.2,0.4,0.6,0.8,1.0,None]"
    - None (désactive le sous-échantillonnage)

## Résultats

Les résultats de la reconstruction d'id **config_id** avec les logs sont sauvegardés dans le dossier **PATH_DOCUMENTS_FOLDER/jobs/config_id/**.

## Architecture

Les principaux fichiers sont :

- [end_to_end_architecture_volume.py](/reconstruction_with_dl/end_to_end_architecture_volume.py) : entraînement bout à bout de Fluofire, il contient deux fonctions de train, **train** pour l'entraînement standard de Fluofire et **train_with_hps** pour l'entraînement avec la méthode HPS

- [run.py](/stage_gaetan/run.py) : script de lancement des reconstructions sur le cluster de calcul selon les configurations définies dans [configs.csv](/stage_gaetan/configs.csv)

- [configs.csv](/stage_gaetan/configs.csv) : configurations des reconstructions, une configuration par ligne

- [main.py](/stage_gaetan/main.py) : définit l’ensemble des hyperparamètres de génération de données et d’apprentissage, puis lance selon les options choisies la génération des données, la reconstruction et le profilage du code

- [generation.py](/stage_gaetan/generation.py) : gère la génération des jeux de données synthétiques

- [reconstruction.py](/stage_gaetan/reconstruction.py) : lance la reconstruction, charge les vues, construit le jeu de données, puis lance l’apprentissage du modèle Fluofire, avec ou sans l’algorithme HPS

- [profiling.py](/stage_gaetan/profiling.py) : lance une reconstruction avec torch.profiler afin de mesurer le temps de calcul CPU/GPU de chaque étape et d’exporter des traces exploitables avec l’outil [Perfetto](https://ui.perfetto.dev/)

- [pose_search.py](/stage_gaetan/pose_search.py) : définit la table de correspondance de poses utilisée pendant l’apprentissage, ainsi que le renvoi vers l’algorithme de HPS ou de SGD selon l’epoch de la reconstruction

- [hps.py](/stage_gaetan/hps.py) : implémente l’algorithme de recherche hiérarchique de pose HPS

> [!NOTE] 
> Pour le moment, HPS n'estime que les rotations et considère des translations nulles

- [hps_visualization.py](/stage_gaetan/hps_visualization.py) : outil de visualisation des itérations de la HPS

- [paths.py](/stage_gaetan/paths.py) : définit les chemins principaux utilisés dans le code

- [utils.py](/stage_gaetan/utils.py) : définit notamment le parser d’arguments et la génération des **config_id**