from snorkel.labeling.model import LabelModel, MajorityLabelVoter
# from abc import ABC, abstractmethod

from src.models.ModelBaseClass import ModelBaseClass
class SnorkelLabelModel(ModelBaseClass):
    def __init__(self, cardinality=2, verbose=True):
        self.model = LabelModel(cardinality=cardinality, verbose=verbose)

    def train(self, L_train, Y_dev=None, n_epochs=500, log_freq=100, seed=123):
        self.model.fit(L_train=L_train, Y_dev=Y_dev, n_epochs=n_epochs, log_freq=log_freq, seed=seed)

    def predict(self, L):
        return self.model.predict(L)

    def report_trained_parameters(self):
        return {
            "weights": self.model.get_weights(),
            # "accuracy": self.model.score(L, Y, metrics=["accuracy"])["accuracy"]
        }


class SnorkelMajorityLabelVoter(ModelBaseClass):
    def __init__(self, cardinality=2):
        self.model = MajorityLabelVoter(cardinality=cardinality)

    def train(self, L_train, Y_train=None):
        # MajorityLabelVoter doesn't have a fit/train method as it's based on majority voting
        pass

    def predict(self, L):
        L_as_int = L.astype(int)
        return self.model.predict(L_as_int)

    def report_trained_parameters(self):
        # MajorityLabelVoter doesn't have parameters to learn
        return "MajorityLabelVoter has no trainable parameters."


# from snorkel.labeling.model import LabelModel

# label_model = LabelModel(cardinality=2, verbose=True)
# label_model.fit(L_train=L_train, n_epochs=500, log_freq=100, seed=123)


# majority_acc = majority_model.score(L=L_test, Y=Y_test, tie_break_policy="random")[
#     "accuracy"
# ]
# print(f"{'Majority Vote Accuracy:':<25} {majority_acc * 100:.1f}%")

# label_model_acc = label_model.score(L=L_test, Y=Y_test, tie_break_policy="random")[
#     "accuracy"
# ]
# print(f"{'Label Model Accuracy:':<25} {label_model_acc * 100:.1f}%")


# >>> L = np.array([[0, 0, -1], [-1, 0, 1], [1, -1, 0]])
# >>> Y_dev = [0, 1, 0]
# >>> label_model = LabelModel(verbose=False)
# >>> label_model.fit(L)
# >>> label_model.fit(L, Y_dev=Y_dev)
# >>> label_model.fit(L, class_balance=[0.7, 0.3])


# >>> L = np.array([[1, 1, 1], [1, 1, -1], [-1, 0, 0], [0, 0, 0]])
# >>> label_model = LabelModel(verbose=False)
# >>> label_model.fit(L, seed=123)
# >>> np.around(label_model.get_weights(), 2)  # doctest: +SKIP
# array([0.99, 0.99, 0.99])



# >>> L = np.array([[0, 0, -1], [1, 1, -1], [0, 0, -1]])
# >>> label_model = LabelModel(verbose=False)
# >>> label_model.fit(L)
# >>> label_model.predict(L)
# array([0, 1, 0])


# >>> L = np.array([[1, 1, -1], [0, 0, -1], [1, 1, -1]])
# >>> label_model = LabelModel(verbose=False)
# >>> label_model.fit(L)
# >>> label_model.score(L, Y=np.array([1, 1, 1]))
# {'accuracy': 0.6666666666666666}
# >>> label_model.score(L, Y=np.array([1, 1, 1]), metrics=["f1"])
# {'f1': 0.8}


# from snorkel.labeling import LabelModel, PandasLFApplier

# # Define the set of labeling functions (LFs)
# lfs = [lf_keyword_bad, lf_keyword_good, lf_keyword_fair]

# # Apply the LFs to the unlabeled training data
# applier = PandasLFApplier(lfs)
# L_train = applier.apply(df_train)

# # Train the label model and compute the training labels
# label_model = LabelModel(cardinality=3, verbose=True)
# label_model.fit(L_train, n_epochs=500, log_freq=50)
# df_train["label"] = label_model.predict(L=L_train, tie_break_policy="abstain")



# from snorkel.labeling import MajorityLabelVoter
# from sklearn.preprocessing import MultiLabelBinarizer
# Y = [['POSITIVE', 'NEGATIVE', 'NEUTRAL']]
# # fit a MultiLabelBinarizer
# mlb = MultiLabelBinarizer()
# mlb.fit_transform(Y)
# # create a majority vote model and predict
# majority_model = MajorityLabelVoter(cardinality=3)
# predictions = majority_model.predict_proba(L=L_test)
# df_multilabel = pd.DataFrame()
# df_multilabel['predict_proba'] = predictions.tolist()
# # get all the non zero indices which are the multi labels
# df_multilabel['multi_labels'] = df_multilabel['predict_proba'].apply(lambda x: np.nonzero(x)[0])
    
# #transform to mlb for classification report
# df_multilabel['mlb_pred'] = df_multilabel['multi_labels'].apply(lambda x: mlb.transform([x])[0])
# print(df_multilabel.head())
# #convert to str in order to see how many multi labels did we gain
# multi_label_string = df_multilabel.multi_labels.apply(lambda x: ", ".join(le.inverse_transform(x)))
# print(multi_label_string.value_counts()[:50])
# # print some metrics using classification report 
# y_pred = df_multilabel.mlb_pred.apply(lambda x: list(x)).to_numpy().tolist()
# y_true = mlb.transform(Y.values).tolist()
# print(classification_report(y_true, y_pred, target_names = mlb.classes_))


# from snorkel.labeling import LabelModel, PandasLFApplier

# # Define the set of labeling functions (LFs)
# lfs = [lf_keyword_bad, lf_keyword_good, lf_keyword_fair]

# # Apply the LFs to the unlabeled training data
# applier = PandasLFApplier(lfs)
# L_train = applier.apply(df_train)

# # Train the label model and compute the training labels
# label_model = LabelModel(cardinality=3, verbose=True)
# label_model.fit(L_train, n_epochs=500, log_freq=50)
# df_train["label"] = label_model.predict(L=L_train, tie_break_policy="abstain")

# label_model.fit(L_train, Y_dev, n_epochs=5000, log_freq=500, seed=12345)
