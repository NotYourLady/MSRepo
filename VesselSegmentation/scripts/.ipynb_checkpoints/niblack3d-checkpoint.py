import numpy as np
import integral_volume as iv

class Niblack3d:
    def __init__(self, window_size=(3, 3, 3), coef_k=0, coef_a=0):
        self.window_size = window_size
        self.coef_k = coef_k
        self.coef_a = coef_a
      
    def binarize(self, vol):
        vol_sq = vol**2

        integr = iv.IntegralVol(vol)
        integr_sq = iv.IntegralVol(vol_sq)

        bin_vol = np.zeros_like(vol, dtype=bool)

        sigmas = []

        for i in range(0, integr.integral_vol.shape[0]-1):
            for j in range(0, integr.integral_vol.shape[1]-1):
                for k in range(0, integr.integral_vol.shape[2]-1):

                    w_s = (self.window_size[0]//2,
                           self.window_size[1]//2,
                           self.window_size[2]//2)

                    f = iv.Point3d(i-w_s[0], j-w_s[1], k-w_s[2]).edit_to_vol(bin_vol)
                    s = iv.Point3d(i+w_s[0], j+w_s[1], k+w_s[2]).edit_to_vol(bin_vol)

                    T, sigma = self.calc_T(integr, integr_sq, w_s, f, s)
                    sigmas.append(sigma)

                    #if (sigma < 120) and (sigma > 80):
                    #    w_s = (w_s[0]*2, w_s[1]*2, w_s[2]*2)
                    #    f = iv.Point3d(i-w_s[0], j-w_s[1], k-w_s[2]).edit_to_vol(bin_vol)
                    #    s = iv.Point3d(i+w_s[0], j+w_s[1], k+w_s[2]).edit_to_vol(bin_vol)
                    #    T, sigma = self.calc_T(integr, integr_sq, w_s, f, s)


                    if vol[i, j, k]>=T:
                        bin_vol[i, j, k] = 1  
        
        return(bin_vol)
    
    def calc_mu(self, integr_vol, p1, p2):
        mu = integr_vol.calculate_sum(p1, p2)
        mu *= (1/self.window_size[0]/self.window_size[1]/self.window_size[2])
        return mu
    
    def calc_T(self, integr, integr_sq, w_s, f, s):
        mu = self.calc_mu(integr, f, s)
        mu_sq = self.calc_mu(integr_sq, f, s)

        sigma = mu_sq - mu**2
        eps = 1e-6
        sigma += eps

        if sigma<0:
            #print("WARNING! sigma<0")
            sigma = 0
        #assert sigma >= 0

        sigma = sigma**0.5
        T = mu + (self.coef_k * sigma) + self.coef_a
        return(T, sigma)
    
class Threser:
    def __init__(self, window_size=(3, 3, 3), threshold):
        self.window_size = window_size
        self.threshold = threshold
    
    
    def binarize_vol(self, vol):
        integr_vol = iv.IntegralVol(vol)
        
        w_s = (self.window_size[0]//2, self.window_size[1]//2, self.window_size[2]//2)
        bin_vol = np.zeros_like(vol, dtype=bool)

        for i in range(0, self.integral_vol.shape[0]-1):
            for j in range(0, self.integral_vol.shape[1]-1):
                for k in range(0, self.integral_vol.shape[2]-1):
                    
                    f = Point3d(i-w_s[0], j-w_s[1], k-w_s[2]).edit_to_vol(bin_vol)
                    s = Point3d(i+w_s[0], j+w_s[1], k+w_s[2]).edit_to_vol(bin_vol)
                    
                    val = integr_vol.calculate_sum(f, s)/self.window_size[0]/self.window_size[1]/self.window_size[2]
                    if val>threshold:
                        bin_vol[i, j, k] = 1
        
        return(bin_vol)