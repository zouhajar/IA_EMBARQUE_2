# IA_EMBARQUE

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

Après avoir généré les fichiers de test (`xtest`, `ytest`) et le modèle au format `.h5`, nous avons réussi à **déployer le modèle sur la carte STM32 sans difficulté**.
L’inférence embarquée a permis d’atteindre une **précision de 85 %** sur le jeu de test.

- **MobileNet ultra-compact** : [Précision ~85 %]  
<p align="center">
  <img src="https://github.com/user-attachments/assets/348594b1-50f2-4c73-a516-d605bbe5fa25" width="500"/>

</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/22862d68-6983-4606-8bd8-642f9fe6c76c" width="500"/>
</p>



