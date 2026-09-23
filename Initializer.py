import dice_ml
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np 


class DICE:

    def __init__(self,model,training_data,features_names,target_ft_name,cf_generation_dice):

        self.model = model
        self.training_data = training_data
        self.target_ft_name = target_ft_name
        data = dice_ml.Data(
            dataframe=training_data,
            continuous_features=features_names,
            outcome_name=target_ft_name,
        )
    
        dice_model = dice_ml.Model(
            model=model,
            backend="sklearn",
        )

        self.exp = dice_ml.Dice(
            data,
            dice_model,
            method=cf_generation_dice,
        )

    def __call__(self, instance, amount_of_cfs):

        if not isinstance(instance, pd.DataFrame):
                instance = instance.to_frame().T

        probabilities = self.model.predict_proba(instance).squeeze()
        predicted_class = np.argmax(probabilities)
        opposite_class = 1 - predicted_class

        opposite_class_cf = self.exp.generate_counterfactuals(
                instance,
                total_CFs=amount_of_cfs,
                desired_class=int(opposite_class),
            )
        
        predicted_class_cf = self.exp.generate_counterfactuals(
                instance,
                total_CFs=amount_of_cfs,
                desired_class=int(predicted_class),
            )

        opposite_class_data = (opposite_class_cf.cf_examples_list[0].final_cfs_df[self.features_names].values)

        predicted_class_data = (predicted_class_cf.cf_examples_list[0].final_cfs_df[self.features_names].values)

        return predicted_class_data, opposite_class_data


class KNeighbors:

    def __init__(self,model,training_data,features_names,algorithm="ball_tree"):

        self.model = model
        self.training_data = training_data
        self.features_names = features_names

        self.nbrs = NearestNeighbors(
                n_neighbors=len(training_data),
                algorithm=algorithm
            ).fit(training_data[features_names])
        
       

    def __call__(self, instance, amount_of_cfs):

        probabilities = self.model.predict_proba(instance).squeeze()
        predicted_class = np.argmax(probabilities)

        _, indices = self.nbrs.kneighbors(instance[self.features_names])

        predicted_class_data = []
        opposite_class_data = []


        for index in indices[0]:
                candidate = self.training_data.iloc[index:index + 1]
        
                candidate_class = self.model.predict(candidate[self.features_names])[0]
        
                if candidate_class == predicted_class:
                    predicted_class_data.append(candidate[self.features_names].squeeze())
                else:
                    opposite_class_data.append(candidate[self.features_names].squeeze())
        
                if (len(predicted_class_data) >= amount_of_cfs and len(opposite_class_data) >= amount_of_cfs):
                    break
        
        predicted_class_data = np.array(predicted_class_data[:amount_of_cfs])
        
        opposite_class_data = np.array(opposite_class_data[:amount_of_cfs])

        return predicted_class_data, opposite_class_data



class RANDOM:

     def __init__(self,model,training_data,features_names,n_samples=1000):

        self.model = model
        self.training_data = training_data
        self.features_names = features_names
        self.n_samples = n_samples

     def __call__(self,instance,amount_of_cfs):

        probabilities = self.model.predict_proba(instance).squeeze()
        predicted_class = np.argmax(probabilities)

        data = np.random.normal(
                      loc=instance[self.features_names].values,
                      scale=self.training_data[self.features_names].std().values,
                      size=(self.n_samples, len(self.features_names)),
                  )

        random_points = pd.DataFrame(data,columns=self.features_names)

        predictions = self.model.predict(random_points)

        predicted_class_data = random_points[predictions == predicted_class].values

        opposite_class_data = random_points[predictions != predicted_class].values

        if len(predicted_class_data) < amount_of_cfs:
            raise ValueError("Not enough same-class random samples.")

        if len(opposite_class_data) < amount_of_cfs:
            raise ValueError("Not enough opposite-class random samples.")

        return predicted_class_data[:amount_of_cfs], opposite_class_data[:amount_of_cfs]

        




