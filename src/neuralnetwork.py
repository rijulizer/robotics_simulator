import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class NeuralNetwork(nn.Module):
    def __init__(self,no_sensors,driving_rule,excitatory_rule,learning_rate,learning_rule):
        super().__init__()
        self.layer1 = nn.Linear(no_sensors+3, 3,False)
        self.layer1_output = torch.rand(3)
        #print(self.layer1.weight.size())
        self.driving_rule = np.reshape(driving_rule,(self.layer1.weight.size(0),self.layer1.weight.size(1)))
        self.excitatory_rule = np.reshape(excitatory_rule,(self.layer1.weight.size(0),self.layer1.weight.size(1)))
        self.learning_rate = np.reshape(learning_rate,(self.layer1.weight.size(0),self.layer1.weight.size(1)))
        self.learning_rule = np.reshape(learning_rule, (self.layer1.weight.size(0),self.layer1.weight.size(1)))
        self.update_rule = [self.update_hebbian_rule,self.update_post_synaptic_rule,self.update_pre_synaptic_rule,self.update_covariance_rule]
        self.activation_rule = [self.activate_rule_driving,self.activate_rule_modulatory]

    def forward(self, x):
        #print("Weights1"+str(self.layer1.weight))
        x = torch.from_numpy(x)
        x = torch.concat((x,self.layer1_output),0)
        self.layer1_output = F.sigmoid(self.layer1(x))
        self.update(x,self.layer1_output)
        output = self.layer1_output[1:]
        return output.detach().numpy()
    
    def update(self,x,output):
        with torch.no_grad():
            for i in range(self.layer1.weight.size(0)):
                for j in range(self.layer1.weight.size(1)):
                    #print(self.learning_rate[i,j])
                    update_fn = self.update_rule[self.learning_rule[i][j]]
                    self.layer1.weight[i][j] = update_fn(lr = self.learning_rate[i][j],w = self.layer1.weight[i][j],x = x[j],y= output[i])
                    self.layer1.weight[i][j] = self.activation_rule[self.driving_rule[i,j]](self.layer1.weight[i][j])
                    #self.layer1.weight[i][j] = max(-1.0,min(1.0,self.layer1.weight[i][j]))
                    if(self.excitatory_rule[i][j]*self.layer1.weight[i][j] < 0):
                        self.layer1.weight[i][j] *= self.excitatory_rule[i][j]
                    #self.layer1.weight[i][j] += self.learning_rate * x[j] * output[i]*(1-self.layer1.weight[i][j])
                    
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
        return torch.add(torch.mul(F.sigmoid(x),0.5),0.5)
    
    def activate_rule_driving(self,x):
        return F.sigmoid(x)