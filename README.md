
# Projet : IA Embarquée sur STM32

**Auteurs :** Yassmina Bara, Hajar Zouggari

> **Note pour le lecteur :**  
> Le projet est organisé en plusieurs branches Git pour faciliter l’accès aux différentes parties :  
> - `Model` : scripts d’entraînement et d’évaluation des modèles d’IA.  
> - `CubeIde` : projet CubeIDE pour l’embarquement sur STM32.   
> - `Sécurité` : expérimentations sur la robustesse face aux attaques Bit‑Flip et mécanismes de protection.  


## Introduction

Ce projet a pour objectif de **développer, optimiser et embarquer un modèle d’intelligence artificielle sur microcontrôleur STM32**.  
L’enjeu principal consiste à **adapter un réseau de neurones conçu pour un environnement de calcul classique (GPU/CPU)** afin qu’il puisse s’exécuter efficacement sur une **plateforme embarquée à ressources limitées** (mémoire, puissance de calcul, consommation énergétique).

## 1. Architecture du modèle

Le modèle est construit avec plusieurs types de couches pour analyser efficacement les images :  
- **Convolutions 3×3** : héritées de VGG, elles extraient les motifs et textures.  
- **ReLU** : activation rapide qui évite que le modèle perde l’information (vanishing gradient).  
- **BatchNormalization** : stabilise et accélère l’entraînement.  
- **Dropout (0,25 – 0,3)** : limite le surapprentissage.  
- **MaxPooling 2×2** : réduit progressivement la taille des images pour simplifier le calcul.  

---

## 2. Vérifications avant l’embarquement

Avant de déployer le modèle sur une carte électronique, il faut vérifier :  
- **Mémoire disponible** : RAM et Flash.  
- **Processeur** : type et puissance pour exécuter le modèle.  
- **Précision des calculs** : formats numériques supportés.  
- **Consommation et vitesse** : respecter les limites d’énergie et le temps de calcul.  
- **Entrées et sorties** : compatibilité avec capteurs et communications.  
- **Logiciels supportés** : bibliothèques nécessaires pour exécuter le modèle.

---

## 3. Compatibilité avec la carte cible
Pour embarquer notre modèle sur la carte, nous avons créé un projet avec CubeMX et déployé le modèle ainsi que la base de données de test. Dans un premier temps, le modèle étant trop volumineux pour la carte, l’analyse initiale a échoué.

<p align="center">
<img src="https://github.com/user-attachments/assets/08788b30-3d5f-4a35-aa47-5e3cd9856ebc" width="400"/>
</p>

Le modèle est **trop volumineux** pour cette carte :  
- Flash nécessaire : 5,14 MiB (la carte n’offre que 2 MiB).  
- RAM nécessaire : 148,56 KiB (la limite est 192 KiB), laissant très peu de marge pour l’exécution.  

Donc ce modèle ne peut pas être embarqué sur cette carte.

## 4. Approches pour intégrer le modèle sur STM32
### Solution 1 : 
Nous avons d’abord effectué la compression MEDIUM du modèle afin de réduire sa taille et son usage mémoire, ce qui permet de l’exécuter sur la carte malgré ses limitations. Une fois l’analyse validée, nous avons généré le code, puis modifié le fichier app_cubeai.c pour spécifier la taille des données CIFAR-10 (32 * 32 *3 *4), définir le timeout à 0xFFFF et indiquer le nombre de classes à 10.

<p align="center">
<img width="428" height="72" alt="analyse apres compression" src="https://github.com/user-attachments/assets/e00eae21-92a5-4246-8adb-79c89f8200a0" />
</p>

Grâce à cette approche, nous avons réussi à embarquer le modèle tout en conservant une très bonne précision

<p align="center">
  <img src="https://github.com/user-attachments/assets/b331f3b7-3c03-40e7-b52b-2bda7d7df26c" width="500"/>
</p>

### Solution 2 : 

Afin de surmonter les contraintes mémoire du STM32 (2 Mo de Flash et 192 KiB de RAM), nous avons opté pour **MobileNet**, un modèle spécifiquement conçu pour les environnements à ressources limitées.
Compromis précision vs taille mémoire

Lors de nos expérimentations, nous avons pu atteindre une précision maximale d’environ 88 % sur CIFAR‑10 avec le modèle complet. Cependant, le modèle dépassait de très peu la limite de mémoire de la carte STM32 (~20 Ko au-dessus), ce qui le rendait impossible à embarquer.

Après plusieurs tentatives d’optimisation, nous avons choisi un MobileNet ultra-compact qui respecte les contraintes mémoire et permet une précision embarquée d’environ 85 %, constituant le meilleur compromis entre taille et performance pour ce microcontrôleur.

Ce choix s’explique par la volonté de disposer d’un réseau **à la fois compact et performant**, capable de s’exécuter efficacement sur un microcontrôleur.
Contrairement à des architectures plus lourdes comme **VGG** ou **ResNet**, qui comportent plusieurs dizaines de millions de paramètres, **MobileNet** offre une structure optimisée permettant de réduire considérablement la taille du modèle tout en conservant une bonne précision.

<p style="font-size:12px; line-height:1.2">

| **Étape** | **Couche** | **Rôle** | **Détails** |
|:--:|:--|:--|:--|
| 1 | Entrée | Reçoit les images CIFAR-10 | 32×32×3 |
| 2 | Conv 3×3 | Détection de motifs simples | 8 filtres – ReLU |
| 3 | Bloc MobileNet 1 | Extraction légère | Depthwise + Pointwise, 8 filtres |
| 4 | Bloc MobileNet 2 | Réduction et ajout de profondeur | Stride 2, 16 filtres |
| 5 | Bloc MobileNet 3 | Extraction de motifs complexes | 24 filtres |
| 6 | Bloc MobileNet 4 | Nouvelle réduction spatiale | Stride 2, 32 filtres |
| 7 | Bloc MobileNet 5 | Raffinement des détails | 48 filtres |
| 8 | Bloc MobileNet 6 | Détection des formes globales | Stride 2, 64 filtres |
| 9 | Global Avg Pooling | Réduction des cartes | - |
| 10 | Dropout | Évite surapprentissage | 0.1 |
| 11 | Dense (Softmax) | Classification finale | 10 classes |

</p>
Après l’entraînement du modèle, nous avons évalué ses performances sur le jeu de test CIFAR-10.
La précision atteinte est affichée, et nous avons généré la matrice de confusion pour visualiser la qualité des prédictions sur chaque classe.


<p align="center">
<img src="https://github.com/user-attachments/assets/3d8d0d9b-b081-4d09-a13a-9d27049548a0"  width="400"/>
</p>


Après avoir généré les fichiers de test (`xtest`, `ytest`) et le modèle au format `.h5`, nous avons réussi à **déployer le modèle sur la carte STM32 sans difficulté**.
L’inférence embarquée a permis d’atteindre une **précision de 85 %** sur le jeu de test.

- **MobileNet ultra-compact** : [Précision ~85 %]  
<p align="center">
  <img src="https://github.com/user-attachments/assets/348594b1-50f2-4c73-a516-d605bbe5fa25" width="500"/>

</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/22862d68-6983-4606-8bd8-642f9fe6c76c" width="500"/>
</p>

# Sécurité — Robustesse face à l’attaque Bit-Flip (BFA)

## Qu’est-ce qu’une attaque Bit-Flip (BFA) ?

L’**attaque Bit-Flip (BFA)** est une technique d’attaque matérielle visant à **altérer les poids d’un réseau de neurones** en modifiant directement certains bits dans la mémoire (généralement DRAM ou Flash).  
Même une inversion de bit peut provoquer une **dégradation importante des performances**, voire amener le modèle à produire des prédictions totalement erronées.

---

## Évaluation sur TinyVGG

Pour le modèle **TinyVGG quantifié sur 8 bits**, entraîné sur **CIFAR-10**, nous avons évalué la robustesse face à l’attaque BFA.  
Deux configurations ont été testées (train+attaque) :  

- **Modèle nominal** : `lr = 0.01`, `clipping_value = 0.0`, `randbet = 0`  
- **Modèle protégé** : `lr = 0.01`, `clipping_value = 0.1`, `randbet = 1`

Après application progressive des bit-flips :  
- Le **modèle protégé** conserve une précision plus stable que le modèle nominal.  
- Les résultats confirment que le **clipping des poids** et l’utilisation de **RandBET** augmentent la robustesse face aux corruptions de poids.

<p align="center">
  <img src="https://github.com/user-attachments/assets/670178f9-0dd3-4fc1-9ad8-e176f94a0c4c" alt="Résultats BFA TinyVGG" width="500"/>
</p>

---
## Évaluation sur MobileNet

Pour notre modèle **MobileNet** structuré en blocs modulaires :  
1. L’architecture a été **transcrite dans `mobilenet_quan.py`**.  
2. L’entraînement a été réalisé selon **quatre configurations** :

| **Configuration** | **Learning rate (lr)** | **Clipping value** | **RandBET** | **Description** |
|:--|:--:|:--:|:--:|:--|
| Modèle nominal | 0.1 | 0.0 | 0 | Référence de base |
| Faible learning rate | 0.01 | 0.0 | 0 | Réduction de la vitesse d’apprentissage |
| Clipping | 0.1 | 0.1 | 0 | Limitation des poids |
| Clipping + RandBET | 0.1 | 0.1 | 1 | Protection combinée renforcée |

Après avoir lancé l’entraînement avec `train_mobilenet.py` et effectué la phase de maintenance :

- Pour chaque configuration du modèle, nous avons appliqué l’attaque **Bit‑Flip (BFA)** à l’aide du script `bfa_mobilenet.py`.  
- L’attaque a été répétée **sur 5 seeds différentes** ['5555', '758', '3666', '4258', '6213'] afin d’évaluer la variabilité due à l’initialisation et obtenir des résultats statistiquement solides.
  
- À chaque exécution (configuration × seed), un fichier **CSV** a été généré contenant l’évolution de l’accuracy en fonction du nombre de bit‑flips (et autres métriques pertinentes).

- Une fois toutes les exécutions terminées, les CSV sont agrégés et analysés avec `printing_tools.py` afin de tracer les **courbes accuracy vs nombre de bit‑flips** permettant de comparer la robustesse des configurations testées.

 
