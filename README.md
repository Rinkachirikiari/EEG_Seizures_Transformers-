# EEG_Seizures_Transformers

Equipe 7 - [...vos noms...], Fabrice Gagnon, Mathis Rezzouk

Université Laval - Québec - Canada 

# Barème

![image](https://github.com/Rinkachirikiari/EEG_Seizures_Transformers-/assets/90961553/cc0f08de-ea42-4624-93c9-f06018a7723e)

# Description du jeu de donnée 

Le jeu de donnée que nous l'allons utiliser a été recueillis dans le cadre du projet collaboratif entre Children's Hospital Boston (CHB) et le Massachusetts Institute of Technology (MIT). 
Ce jeu de donnée a été construit à partir des données de 22 patients (chb01, chb02, chb03,...), ayant entre 9 et 42 enregistrements stockés en .edf (European Data Format) selon le sujet. Dans la plupart des cas, les fichiers .edf contiennent exactement une heure de signaux EEG numérisés, bien que ceux appartenant au cas chb10 durent deux heures et ceux appartenant aux cas chb04, chb06, chb07, chb09 et chb23 durent quatre heures. De manière occasionnelle, les fichiers dans lesquels des crises sont enregistrées peuvent être plus courts.

Tous les signaux ont été échantillonnés à 256 Hz (256 points en 1 seconde) avec une résolution de 16 bits. La plupart des fichiers contiennent 23 signaux EEG (24 ou 26 dans quelques cas). Le système international de positionnement et de nomenclature des électrodes EEG 10-20 a été utilisé pour ces enregistrements. Dans quelques enregistrements, d'autres signaux sont également enregistrés, tels qu'un signal ECG dans les 36 derniers fichiers appartenant au cas chb04 et un signal de stimulation du nerf vague (VNS) dans les 18 derniers fichiers appartenant au cas chb09. Dans certains cas, jusqu'à 5 signaux "factices" (nommés "-") ont été intercalés parmi les signaux EEG pour obtenir un format d'affichage facile à lire ; ces signaux factices peuvent être ignorés.,

Par channels, nous nous référons aux binômes de capteurs, aux électrodes qui vont être utilisées pour calculer la différence de potentiel entre les différentes zones d’intérêt du cerveau.

Enfin, le fichier RECORDS contient une liste des 664 fichiers .edf inclus dans cette collection, tandis que le fichier RECORDS-WITH-SEIZURES répertorie les 129 de ces fichiers contenant une ou plusieurs crises. Le fichier SUBJECT-INFO contient le genre et l'âge de chaque sujet. (Le cas chb24 a été ajouté à cette collection en décembre 2010 et n'est pas actuellement inclus dans SUBJECT-INFO.)

Dans l'ensemble, ces enregistrements comprennent 198 crises au total ; le début ([) et la fin (]) de chaque crise sont annotés dans les fichiers d'annotations .seizure qui accompagnent chacun des fichiers répertoriés dans RECORDS-WITH-SEIZURES. De plus, les fichiers nommés chbnn-summary.txt contiennent des informations sur le montage utilisé pour chaque enregistrement, ainsi que le temps écoulé en secondes depuis le début de chaque fichier .edf jusqu'au début et à la fin de chaque crise qu'il contient.

