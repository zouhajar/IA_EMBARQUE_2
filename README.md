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
<img src="https://github.com/user-attachments/assets/08788b30-3d5f-4a35-aa47-5e3cd9856ebc" width="400"/>


Le modèle est **trop volumineux** pour cette carte :  
- Flash nécessaire : 5,14 MiB (la carte n’offre que 2 MiB).  
- RAM nécessaire : 148,56 KiB (la limite est 192 KiB), laissant très peu de marge pour l’exécution.  

Donc ce modèle ne peut pas être embarqué sur cette carte.
