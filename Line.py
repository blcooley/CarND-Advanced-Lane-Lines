import numpy as np

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # max values in the average
        self.min_samp = 1000
        # x and y values of the last n fits of the line
        self.recent_xyfitted = []
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = np.array([False])
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None

    def sanity_check(self, fit):
        if self.best_fit is None:
            return True

        self.diffs = self.best_fit * fit
        # if (self.diffs < 0.0)[:2].any():
        #    return False

        if len(self.allx) < 100:
            return False
        
        return True
        
        
    def update(self, fit, allx, ally):
        if self.best_fit is None:
            self.best_fit = fit
        else:
            self.best_fit = self.best_fit*0.8 + 0.2*fit            
                    
        self.allx = allx
        self.ally = ally
        self.recent_xyfitted = [(self.allx, self.ally)]

        
