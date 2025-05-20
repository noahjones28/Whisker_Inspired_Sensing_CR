import numpy as np
from smt.surrogate_models import KRG, RBF, KPLSK
from smt.sampling_methods import LHS
from joblib import Parallel, delayed
import glob, os

class MySurrogateModel:
    def __init__(self, method):
            # Initialize the RBF model
            if method == "RBF":
                self.model = RBF(d0=2, poly_degree=1, reg=1e-25)
            elif method == "KRG":
                self.model = KRG()
            elif method == "KPLSK":
                self.model = KPLSK(
                    n_comp=8,  # Improves speed & generalization
                    nugget=1e-4,
                    theta0=[1e-1],  # Better than 0.01
                    hyper_opt="Cobyla",  # Faster and more stable for high-D problems
                )
            else:
                 raise ValueError("Invalid surrogate model type!")
            
    def get_training_data(self, F_limits, s_limits, tendon_displacement_limits, n_training_samples, elastica_model, wrist_model):
        limits = np.array([[F_limits[0], F_limits[-1]], [s_limits[0], s_limits[-1]], [tendon_displacement_limits[0], tendon_displacement_limits[-1]]])
        n_testing_samples = int(np.round(0.2*n_training_samples))
        sampling = LHS(xlimits=limits, criterion='maximin')
        # Training Samples
        training_samples = sampling(n_training_samples)  # Generate initial training points
        F_training_samples = np.array(training_samples[:, 0]) # get all sampeld F's
        s_training_samples = np.array(training_samples[:, 1]) # get all sampeld s's
        tendon_displacement_training_samples = np.array(training_samples[:, 2])
        x_train = np.column_stack((F_training_samples, s_training_samples, tendon_displacement_training_samples))
        y_train = Parallel(n_jobs=-1)(
            delayed(elastica_model.getProximalValues)(F=row[0], s=row[1], tendon_displacement=row[2], wrist_model=wrist_model) for row in x_train
        )
        # Testing Samples
        testing_samples = sampling(n_testing_samples)  # Generate initial testing points
        F_testing_samples = np.array(testing_samples[:, 0]) # get all sampeld F's
        s_testing_samples = np.array(testing_samples[:, 1]) # get all sampeld s's
        tendon_displacement_testing_samples = np.array(testing_samples[:, 2])
        x_test = np.column_stack((F_testing_samples, s_testing_samples, tendon_displacement_testing_samples))
        y_test = Parallel(n_jobs=-1)(
            delayed(elastica_model.getProximalValues)(F=row[0], s=row[1], tendon_displacement=row[2], wrist_model=wrist_model) for row in x_test
        )
        # Save the scaled train and test sets as .npy
        os.makedirs("distal_to_proximal_training_data", exist_ok=True)
        np.save("distal_to_proximal_training_data/x_train.npy", x_train)
        np.save("distal_to_proximal_training_data/y_train.npy", y_train)
        np.save("distal_to_proximal_training_data/x_test.npy", x_test)
        np.save("distal_to_proximal_training_data/y_test.npy", y_test)
        raise ValueError("Sampling calculations complete! Please re-run code")
    
    def train_model(self):
        # Load training data
        x_train = np.load("distal_to_proximal_training_data/x_train.npy")
        y_train = np.load("distal_to_proximal_training_data/y_train.npy")
        # Load testing data
        x_test = np.load("distal_to_proximal_training_data/x_test.npy")
        y_test = np.load("distal_to_proximal_training_data/y_test.npy")
        # Train the model
        self.model.set_training_values(x_train, y_train)
        self.model.train()
        # Train set error
        y_train_pred, y_train_std = self.model.predict_values(x_train), self.model.predict_variances(x_train)
        train_error = np.linalg.norm(y_train - y_train_pred) / np.linalg.norm(y_train)
        print("\ntrain error:", train_error)
        # Test set error
        y_test_pred, y_test_std = self.model.predict_values(x_test), self.model.predict_variances(x_test)
        test_error = np.linalg.norm(y_test - y_test_pred) / np.linalg.norm(y_test)
        print("\ntest error:", test_error)
    
    def predict(self, x):
        prediction = self.model.predict_values(x)
        return prediction