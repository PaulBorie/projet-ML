#!/usr/bin/env python3

import numpy as np
from keras.datasets import mnist


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train=x_train.reshape(60000, 784)
x_test=x_test.reshape(10000,784)
x_test_small = x_test[0:200, :]
y_test_small = y_test[0:200]

class Bayes:
    
    ## Gaussian Naive Bayes

    def __init__(self, var_noise):
        self.var_noise = var_noise

    def train(self):
        self.fit(x_train, y_train)
        

    def fit(self, x_train, y_train):

        # on redimenssionne les features en (784, 1) poru qu'il soit 2 dimensions et qu'on puisse le concaténer à ces features
        two_dim_y_train = y_train.reshape((y_train.shape[0], 1))

        # on concatene les features avec les target
        concat_feature_target = np.hstack((x_train, two_dim_y_train))
        print("Full dataset with feature and target")
        print(concat_feature_target)
        print("\n")
        
        # on initialise un tableau qui acceuillera les moyennes de chaque feature pour chaque classe (chaque ligne sera une classe et les colonnes sera une feature )
        self.all_class_means = np.empty((0,x_train.shape[1])) 

        # on initialise un tableau qui acceuillera les variances de chaque feature pour chaque classe (chaque ligne sera une classe et les colonnes sera une feature )
        self.all_class_vars = np.empty((0, x_train.shape[1]))

        # On récupère toutes les classes dans les target
        self.classes = np.unique(y_train)
        print("Les différentes classes du Dataset")    
        print(self.classes)
        self.nb_class = self.classes.shape[0]
        print("Le nombre de classes {}".format(self.nb_class))

        # on itilialise un tablea qui acceuillera la probabilité pour chaque classe
        self.probas_classes = np.empty((0, 1))

        for c in self.classes:

            print("\n////////////////////\n")
            print("Calcul de la moyenne et de l'écart type pour chaque feature de la classe: {}\n".format(c))  

            # On fait un mask pour récupérer un sous-dataset avec que les données d'une classe c
            self.nb_features = x_train.shape[1] 
            mask = (concat_feature_target[:, self.nb_features] == c)
            specific_class = concat_feature_target[mask, :]
            print("Sous ensemble du dataset avec toutes les datas de la classe: {}".format(c))
            print(specific_class)
            print("\n")

            # On calcule la probabilité p(class) = count(datas de la class C) / count(all dataset) pour la classe courante c 
            proba_class = specific_class.shape[0] / x_train.shape[0]
            
            self.probas_classes = np.append(self.probas_classes, [[proba_class]], axis=0)
            print("Proba pour la classe {}".format(c))
            print(proba_class)
            print("\n")

            # On calcule la moyenne pour chaque colonne, cad pour chaque feature 
            means = specific_class.mean(axis=0)
            # On enleve le dernier élément des moyennes car c'est la moyenne de la colonne target qui ne sert à rien
            means = np.delete(means, self.nb_features, 0)
            twodim_means = means.reshape((1, means.shape[0]))
            print("Moyenne pour chaque feature de la classe {}".format(c))
            print(twodim_means)
            
            print("\n")
            # On append à All_target_means qui contient pour chaque classe la moyenne de chaque feature (chaque ligne est une classe et chaque colone une feature)
            self.all_class_means = np.append(self.all_class_means, twodim_means , axis=0)
           
            vars = specific_class.var(axis=0) 
            vars=vars+self.var_noise #On ajoute du noise à la variance pour éviter les cas ou elle est égale à 0 et que la formule ne peut pas aboutir car division par 0
            vars = np.delete(vars, self.nb_features, 0)
            twodim_vars = vars.reshape((1, vars.shape[0]))
            print("L'écart-type pour chaque feature de la classe {}".format(c))
            print(twodim_vars)
            print("\n")
            # On append à All_target_vars qui contient pour chaque classe la variance de chaque feature une ligne avec la variance pour chaque feature de la classe en cours de traitement c
            self.all_class_vars = np.append(self.all_class_vars, twodim_vars, axis=0)

        print("RES FINAL DE L'ENTRAINEMENT")
        print("\nLes probas pour chaque classe\n")
        print(self.probas_classes)
        print("\nLes moyennes pour chaque classe\n")
        print(self.all_class_means)
        print("\nLa variance pour chaque classe\n")
        print(self.all_class_vars)

      
    def predict(self, x_test, y_test):
        nb_test_data = x_test.shape[0]
        all_pred = []
        for i in range(x_test.shape[0]):
            all_class_prob = []
            for class_index in range(self.nb_class): 
                numerator=np.exp(-((x_test[i]-self.all_class_means[class_index])**2)/(2* self.all_class_vars[class_index])) 
                denominator=np.sqrt(2*np.pi*(self.all_class_vars[class_index]))
                probas_each_feature=numerator/denominator # Dans ce tableau on a la proba pour chaque feature maintenant il faut les multiplier entre elles et multiplier par la proba p(class)
                #print("classe {}".format(self.classes[class_index]))
                #print("p(class) {}".format(self.probas_classes[class_index]))

                #print("proba each feature: {}".format(probas_each_feature))
                #log(ab)=log(a)+log(b)
                # https://stats.stackexchange.com/questions/163088/how-to-use-log-probabilities-for-gaussian-naive-bayes
                # log(P(class i| data))∝log(P(classi))+∑jlog(P(dataj|classi))
                proba_class=np.sum(np.log(probas_each_feature))  # On applique le logarithme car les valeurs sont trop petites
                proba_class = proba_class +  np.log(self.probas_classes[class_index])
                all_class_prob.append(proba_class) 
                #print("\n")
            # On a fini de calculer la probabilité pour chaque classe    
            max_class_index=all_class_prob.index(max(all_class_prob))
            pred = self.classes[max_class_index]
            #print("class prediction: {}".format(pred))
            all_pred.append(pred)
            #print("all pred: {}".format(all_pred))
            good_pred_count = np.count_nonzero(all_pred == y_test)
            self.accuracy = good_pred_count / nb_test_data
            #print("accuracy: {}".format(accuracy)) 

        return all_pred, self.accuracy

    def score(self):
        return self.accuracy

        
def main():  
    bayes = Bayes()
    train_x = np.array([[4, 1, 2], [5, 3, 1], [8, 8, 2], [9, 9, 1], [2, 3, 4]])
    train_y = np.array([3, 3, 1, 1, 4])

    test_x = np.array([[2, 3, 4]])
    test_y = np.array([4])



    bayes.fit(train_x, train_y)

    print("\n////////////// TEST ////////////// \n")
    print(train_x.shape)
    print(train_y.shape)
    bayes.predict(test_x, test_y)

if __name__ == "__main__":  
    main()