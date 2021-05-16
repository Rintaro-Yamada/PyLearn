import numpy as np

class RBFkernel():
    def __init__(self,length_scale, variance, noise_var):
        self.length_scale = length_scale
        self.variance = variance
        self.noise_var = noise_var
    
    def K(self, X1, X2):
        norm = X1 ** 2 - 2 * np.dot(X1, X2.T) + X2.T ** 2
        return self.variance * np.exp(- norm / (2 * self.length_scale ** 2))
        
    def predict(self, kernel, X, X_train, y_train):
        K_inv = np.linalg.inv(kernel.K(X_train, X_train)+ self.noise_var*np.eye(X_train.shape[0]))
        k_star = kernel.K(X_train, X)
        k_star_star = np.full_like(X, self.variance)
        k_star_trans_K_inv = np.dot(k_star.T, K_inv) #k_*.T K^{-1}
        pred_mean = np.dot(k_star_trans_K_inv, y_train)
        pred_var = k_star_star - np.diag((np.dot(k_star_trans_K_inv, k_star))).reshape(X.shape[0], 1)
        #einsumを使った実装
        '''
        pred_var = k_star_star - np.einsum("ij,ji->ij",k_star_trans_K_inv, k_star)
        print(pred_var)
        sys.exit()
        '''
        return pred_mean, pred_var

