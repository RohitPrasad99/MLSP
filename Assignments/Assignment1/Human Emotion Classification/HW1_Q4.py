"""
Created on Fri Feb  4 18:35:01 2022

@author: Rohit
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg
from tqdm import tqdm


class PCA_LDA:
    # class variable
    flag = 0
    
    def __init__(self):
        pass
    
    
    def DecompositionLargeDim(self,x):
        '''
        Computes Eigen Decomposition of data Covariance Matrix
        
        args:
        -----
        x : a (N x D) data matrix
      
        return:
        ------
        None
        '''
        N = x.shape[0]
        # print("In Decomp",x.shape,N)
        self.x_bar = np.mean(x,axis = 0).reshape(1,-1)
        x = x - self.x_bar
        # print(x.shape)
        cov_x = 1/N * (x @ x.T)
        # print("cov_x",np.linalg.det(cov_x),"  rank ->",np.linalg.matrix_rank(cov_x),cov_x.shape)
        self.eig_val,eig_vec = np.linalg.eigh(cov_x)
        self.eig_val = self.eig_val[::-1]
        eig_vec = eig_vec[:,::-1]
      
        # Filtering non zero eigen vectors and eigen values
        # 
        eig_vec = eig_vec[:,self.eig_val > 10**-10]
        self.eig_val = self.eig_val[self.eig_val > 10**-10]
        V = x.T @ eig_vec
        W = np.diag(1 /np.sqrt(N * self.eig_val))  
        self.eig_vec =  V @ W
    
    def one_shot_PCA(self,x,k):
        '''
        DO PCA ON DATA - Calls DecompositionLargeDim() once to compute eigen vectors of covariance matrix
        then takes 1st k eigen vectors and does projection on those vectors
        
        args:
        -----
        x : a (N x D) data matrix 
      
        return:
        -------
        red_x : a (N x k) reduced dimensional data matrix
        '''
        if self.flag == 0:
            self.DecompositionLargeDim(x)
            self.flag = 1
        self.U = self.eig_vec[:,:k] #np.array(X)
        self.red_matrix = (x - self.x_bar)  @ self.U
        self.k = k
        return self.red_matrix
      
    def reconstruct(self,x):
        '''
        to reconstruct a (N x D) data matrix from a (N x k) reduced dimensional data matrix
      
        args:
        -----
        x = a (N x k) reduced dimensional data matrix
      
        returns:
        -------
        x = a (N x D) reconstructed data matrix 
        '''
        x = x @ self.U.T
        # print(x.shape,self.x_bar.shape)
        x = x + self.x_bar.reshape(1,-1)
        return x
    
    def LDA2(self,x,y,mat_req = 0):
        '''
        to perform LDA on a (N x k) data matrix to project onto 1 dimension
      
        args:
        -----
        x       = a (N x k) reduced dimensional data matrix obtained from PCA
        y       = a (N x 1) class label array (y = 1 ---> happy) and (y = 0 ---> sad)
        mat_req = if 1, computes a (N x 1) data matrix reduced to 1 dimension (defaults to 0) for each class
      
        Returns:
        -------
        fin_mat_c1   = a (s x 1) reconstructed data matrix for class label c1 if no of data with y=c1 is s
        fin_mat_c1   = a ((N - s) x 1) reconstructed data matrix for class label c2 
        self.f_ratio = Fisher Ratio for the projected data 
        self.W       = a (k x 1) data array used to do LDA transformation
        '''
        c1 = []
        c2 = []
        c1 = x[y == 0,:]
        c2 = x[y == 1,:]
        
        c1_mean = np.mean(c1,axis = 0).reshape(-1,1)
        c2_mean = np.mean(c2,axis = 0).reshape(-1,1)
        # print(c1_mean.shape,c2_mean.shape)
        
        self.S_b = (c1_mean - c2_mean) @ (c1_mean - c2_mean).T
        
        N1 = c1.shape[0]
        N2 = c2.shape[0]
        # print(N1,N2)
        a = (c1.T - c1_mean)
        b = (c2.T - c2_mean)
      
        # print(a.shape,b.shape)
        
        c1_var = 1/N1 * (a @ a.T)
        c2_var = 1/N2 * (b @ b.T)
      
        self.S_w = c1_var + c2_var
        
        # print(np.linalg.matrix_rank(c1_var),np.linalg.matrix_rank(c2_var),x.shape[1],np.linalg.matrix_rank(self.S_w))
        
        S_w_inv = np.linalg.inv(self.S_w)
      
        F = S_w_inv @ self.S_b
         
        e_val,e_vec = np.linalg.eig(F)
      
        # sorted_idx_e_val = np.argsort(e_val)
        # e_val = 
        
        max_index = np.argmax(e_val)
      
        W = e_vec[:,max_index].reshape(-1,1)
        
        # print(x.shape,e_val)
        # plt.figure()
        # plt.plot(self.W)
        # plt.show()
        
        fin_mat_c1 = 0
        fin_mat_c2 = 1
        if mat_req == 1:
            fin_mat_c1 = c1 @ W 
            fin_mat_c2 = c2 @ W 
      
        # print(c1_mean.shape,c2_mean.shape)
        # print(W.shape)
      
        self.calculate_F_ratio(W)
        return fin_mat_c1,fin_mat_c2,self.f_ratio,W
    
    def calculate_F_ratio(self,W):
        '''
        Calculates Fisher Ratio for the particular transformation array W
      
        '''
        a = W.T @ (self.S_b @ W)
        b = W.T @ (self.S_w @ W)
        self.f_ratio =  a/b
        return self.f_ratio
    
    def compute_optimal_k(self,x_train,y_train):
        '''
        Function to find best value of k such that when we do dim reduction by PCA from (NxD) to (Nxk)
        and by LDA from (Nxk) to (Nx1) our F-Ratio is maximized.
      
        args:
        -----
        x_train : a (N x D) training data where each data is (D X 1) image data
        y_train : a (Nx1) training label
      
      
        returns:
        -------
        k          : best value for such k 
        ratio_list : Fisher Ratio list for for all values of k = [1,min(N,D)] 
        '''
        self.flag = 0
        self.k = -1
        mx = -1
        ratio_list = []
        for i in tqdm(range(1,min(x_train.shape)-1,1)):
            red_x = self.one_shot_PCA(x_train,i)
            _,_,ratio,W = self.LDA2(red_x,y_train)
            ratio_list.append(np.squeeze(ratio))
            if ratio > mx:
                mx = ratio
                self.k = i
                self.W = W
        
        return self.k,ratio_list
    
    def plot_best_f_ratio(self,ratio_list):
        '''
        plots the F-Ratio for all values of k
      
        args:
        ratio_list : F-Ratio list obtained in compute_optimal_k() 
        '''
        plt.figure(figsize=(10,8))
        plt.title("Fisher Ratio for different values of k")
        plt.ylabel("Fisher Ratio")
        plt.xlabel(" k -->")
        plt.plot(np.arange(1,len(ratio_list)+1),ratio_list, "*-" ,color = 'r')
        plt.xticks(np.arange(1,len(ratio_list)+1))
        plt.grid()
        plt.show()
    
    def projection(self,x,y):
        '''
        does projection of input data matrix (NxD) transforming it into (Nxk) first by PCA
        then (Nx1) by LDA where parameters of PCA and LDA are for optimal value of k

        Parameters
        ----------
        x : (NxD)
            input data matrix.
        y : (Nx1)
            class label of input matrix.

        Returns
        -------
        vals : (Nx1)
            projected data matrix.

        '''
        N = x.shape[0]
        x = x.reshape((N,-1))
        vals = ((x - self.x_bar) @ self.U) @ self.W
        return vals
    
    def compute_threshold(self,x,y,vals):
        '''
        Computes best value of threshold which seperates the two classes based 
        on weighted mean of two classes

        Parameters
        ----------
        x : (NxD)
            input data matrix.
        y : (Nx1)
            class label of input matrix.
        vals : (Nx1)
            projected data matrix based on optimal value of k selected in LDA2.

        Returns
        -------
        None.

        '''
        # print(vals.shape,N,y.shape,np.arange(1,N+1).shape)
        
        vals_c1 = vals[y==0]
        vals_c2 = vals[y==1]
        
        vals_c1_mean = np.mean(vals_c1)
        vals_c2_mean = np.mean(vals_c2)
        
        N1 = vals_c1.shape[0]
        N2 = vals_c2.shape[0]
        
        # weighted average of within class mean to find threshold
        self.threshold = np.abs(((N1 * vals_c1_mean) + (N2 * vals_c2_mean)) / (N1 + N2))
        
    def plot_class_seperation(self,x,y,vals,var = "test"):
        '''
          plots input data after doing combined PCA-LDA Transformation
          
          Parameters
          ----------
          x : (NXD) 
              input data matrix.
          y : (NX1)
              Label of input data.
          vals : (NX1)
              transformation data matrix after doing LDA.
          var : string, optional
              on which data - train or test accuracy is calculated. The default is "test".

          Returns
          -------
          None.
        
          '''
        vals_c1 = vals[y==0]
        vals_c2 = vals[y==1]
        
        N1 = vals_c1.shape[0]
        N2 = vals_c2.shape[0]
        
        plt.figure()
        plt.title(var + " Data Plot for all data-points for k = " + str(self.k))
        plt.xlabel("Data points")
        plt.ylabel("Projection value of dataPoint after PCA-LDA operation")
        plt.scatter(np.arange(1,N1+1),vals_c1,color = 'yellow')
        plt.scatter(np.arange(1,N2+1),vals_c2,color = 'black')
        plt.legend(["sad","happy"])
        plt.hlines(xmin = 0,xmax = (max(vals_c1.shape[0],vals_c2.shape[0])+1),y = self.threshold)
        plt.show()
    
      
    def find_accuracy(self,x,y,vals,var="test"):
        '''
        Calculates the accuracy of prediction the class of images based on threshold

        Parameters
        ----------
        x : (NxD) array
            input (NxD) array each N = no of input images.
        y : (Nx1) array
            class label of input array.
        vals : (Nx1) array
            projected data matrix after applying PCA and LDA
        var : string, optional
            on which data - train or test accuracy is calculated. The default is "test".

        Returns
        -------
        None.

        ''' 
        y_hat = np.array(vals < self.threshold).astype(np.int32).squeeze()
        # print(y_hat)
        acc = (np.sum(y == y_hat) / y.shape[0] )* 100
        print('#'*50)
        print("The accuracy on ",var," is ", acc,"% for the k = ",self.k , " and threshold value y = ", self.threshold)
        print('#'*50)

def read_data(var):
    '''
      Reads Image Data from file
    
      Parameters
      ----------
      var : string
          "train" to read training Images.
          "test" to read testing Images.
    
      Returns
      -------
      x : (Nxaxb) numpy array
          N images each of size (axb).
      y : (Nx1) numpy array
          labels for each images.
    
      '''
    data_dir="G:\\IISc\\MLSP\\Assignment 1\\HW-1 Q-4\\Data\\emotion_classification\\"
    train_dir = data_dir+var+"/"
    train_img_paths= os.listdir(train_dir)
    x = []
    y = []
    
    for im in train_img_paths:
      y.append(int(bool(im.find("sad")<0)))
      x.append(mpimg.imread(train_dir+im))
      # print(x[-1].shape)
    y = np.asarray(y)
    x=np.asarray(x).astype(np.float64)
    x = x.reshape(x.shape[0],-1)
    return x,y
    # x.shape



'''
Actual Main code
'''
if __name__ == "__main__":
    x_train,y_train=read_data("train")
    print(x_train.shape,y_train.shape)
    obj = PCA_LDA()
    # computes optimal k value and f_ratio list by doing PCA -LDA
    best_k,f_ratio = obj.compute_optimal_k(x_train,y_train)
    
    # x = x.reshape(20,-1)
    # red_max = obj.one_shot_PCA(x, 18)
    # _,_,_,W = obj.LDA2(red_max,y)
    
    x_test,y_test = read_data("test")
    x_test = x_test.reshape(x_test.shape[0],-1)
    
    # Projecting onto 1 dimension
    x_train_projection = obj.projection(x_train, y_train)
    x_test_projection = obj.projection(x_test, y_test)
    
    # Computing best threshold 
    obj.compute_threshold(x_train, y_train, x_train_projection)
    
    # Plotting Fisher Ratio at different values of k
    obj.plot_best_f_ratio(f_ratio)
    
    # plotting data alongwith threshold
    obj.plot_class_seperation(x_train,y_train,x_train_projection,"train")
    obj.plot_class_seperation(x_test,y_test,x_test_projection)
    
    # find accuracy of classifier based on threshold
    obj.find_accuracy(x_train, y_train,x_train_projection,"train")
    obj.find_accuracy(x_test, y_test,x_test_projection)
    
