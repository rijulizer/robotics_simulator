import numpy as np

class NeuralNetwork2:
    def __init__(self,no_sensors,driving_rule,excitatory_rule,learning_rate,learning_rule):
        self.layer1d = np.random.rand(no_sensors+4, 4).T
        self.layer2d = np.random.rand(4, 2).T
        self.layer1d_output = np.random.rand(4)
        l1_dim = self.layer1d.shape[0]*self.layer1d.shape[1]
        self.driving_rule1 = np.reshape(driving_rule[:l1_dim],(self.layer1d.shape[0],self.layer1d.shape[1]))
        self.excitatory_rule1 = np.reshape(excitatory_rule[:l1_dim],(self.layer1d.shape[0],self.layer1d.shape[1]))
        self.learning_rate1 = np.reshape(learning_rate[:l1_dim],(self.layer1d.shape[0],self.layer1d.shape[1]))
        self.learning_rule1 = np.reshape(learning_rule[:l1_dim], (self.layer1d.shape[0],self.layer1d.shape[1]))
        self.driving_rule2 = np.reshape(driving_rule[l1_dim:],(self.layer2d.shape[0],self.layer2d.shape[1]))
        self.excitatory_rule2 = np.reshape(excitatory_rule[l1_dim:],(self.layer2d.shape[0],self.layer2d.shape[1]))
        self.learning_rate2 = np.reshape(learning_rate[l1_dim:],(self.layer2d.shape[0],self.layer2d.shape[1]))
        self.learning_rule2 = np.reshape(learning_rule[l1_dim:], (self.layer2d.shape[0],self.layer2d.shape[1]))
        self.update_rule = [self.update_hebbian_rule,self.update_post_synaptic_rule,self.update_pre_synaptic_rule,self.update_covariance_rule]
        self.activation_rule = [self.activate_rule_driving,self.activate_rule_modulatory]
        self.max_sensor_input = []

    def forward(self, x):
        self.max_sensor_input.append(np.max(x))
        x1 = np.concatenate((x,self.layer1d_output),0)
        self.layer1d_output = self.sigmoid(self.layer1d@x1)
        output = self.sigmoid(self.layer2d@self.layer1d_output)
        self.update(x1,self.layer1d_output,output)
        return output
    
    def update(self,x,intermediate,output):
        for i in range(self.layer1d.shape[0]):
            for j in range(self.layer1d.shape[1]):
                update_fn = self.update_rule[self.learning_rule1[i][j]]
                #self.layer1d[i][j] = np.abs(self.layer1d[i][j])
                self.layer1d[i][j] = update_fn(lr = self.learning_rate1[i][j],w = self.layer1d[i][j],x = x[j],y= intermediate[i])
                self.layer1d[i][j] = self.activation_rule[self.driving_rule1[i,j]](self.layer1d[i][j])
                #self.layer1d[i][j] = max(0,min(1.0,self.layer1d[i][j]))
                if(self.excitatory_rule1[i][j]*self.layer1d[i][j] < 0):
                    self.layer1d[i][j] *= self.excitatory_rule1[i][j]

        for i in range(self.layer2d.shape[0]):
            for j in range(self.layer2d.shape[1]):
                update_fn = self.update_rule[self.learning_rule2[i][j]]
                #self.layer2d[i][j] = np.abs(self.layer2d[i][j])
                self.layer2d[i][j] = update_fn(lr = self.learning_rate2[i][j],w = self.layer2d[i][j],x = intermediate[j],y= output[i])
                self.layer2d[i][j] = self.activation_rule[self.driving_rule2[i,j]](self.layer2d[i][j])
                #self.layer2d[i][j] = max(0,min(1.0,self.layer2d[i][j]))
                if(self.excitatory_rule2[i][j]*self.layer2d[i][j] < 0):
                    self.layer2d[i][j] *= self.excitatory_rule2[i][j]
                    
    def update_hebbian_rule(self,lr,w,x,y):
        return lr*x*y*(1-w)
    
    def update_post_synaptic_rule(self,lr,w,x,y):
        return lr*(w*(-1+x)*y + (1-w)*x*y)
    
    def update_pre_synaptic_rule(self,lr,w,x,y):
        return lr*(w*x*(-1+y)+ (1-w)*x*y)
    
    def update_covariance_rule(self,lr,w,x,y):
        delta = np.tanh(4*(1-abs(x-y)-2))
        if delta > 0:
            delta *= (1-w)
        else:
            delta *= w
        return lr*delta
    
    def activate_rule_modulatory(self,x):
        return (self.sigmoid(x)*0.25)+0.75
    
    def activate_rule_driving(self,x):
        return self.sigmoid(x)
    
    def sigmoid(self,Z):
        return 1/(1+(np.exp((-Z))))