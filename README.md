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
