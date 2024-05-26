import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class NeuralNetwork(nn.Module):
    def __init__(self,no_sensors,driving_rule,excitatory_rule,learning_rate,learning_rule):
        super().__init__()
        #self.layer1 = nn.Linear(no_sensors+3, 3,False)
        self.layer1d_output = torch.rand(3)
        self.layer1m_output = torch.rand(3)
        self.layer1d = nn.Linear(no_sensors+3, 3,False)
        self.layer1m = nn.Linear(no_sensors+3, 3,False)
        #print(self.layer1.weight.size())
        self.driving_rule = np.reshape(driving_rule,(self.layer1d.weight.size(0),self.layer1d.weight.size(1)))
        self.excitatory_rule = np.reshape(excitatory_rule,(self.layer1d.weight.size(0),self.layer1d.weight.size(1)))
        self.learning_rate = np.reshape(learning_rate,(self.layer1d.weight.size(0),self.layer1d.weight.size(1)))
        self.learning_rule = np.reshape(learning_rule, (self.layer1d.weight.size(0),self.layer1d.weight.size(1)))
        self.update_rule = [self.update_hebbian_rule,self.update_post_synaptic_rule,self.update_pre_synaptic_rule,self.update_covariance_rule]
        self.activation_rule = [self.activate_rule_driving,self.activate_rule_modulatory]
        self.max_sensor_input = []

    def forward(self, x):
        #print("Weights1"+str(self.layer1.weight))
        self.max_sensor_input.append(np.max(x))
        x = torch.from_numpy(x)
        x1 = torch.concat((x,self.layer1d_output),0)
        x2 = torch.concat((x,self.layer1m_output),0)
        self.layer1d_output = self.activate_rule_driving(self.layer1d(x1))
        self.layer1m_output = self.activate_rule_modulatory(self.layer1m(x2))
        self.update(x1,x2,self.layer1d_output,self.layer1m_output)
        output = self.layer1d_output*self.layer1d_output
        #self.layer1_output = F.sigmoid(self.layer1(x))
        #self.update(x,self.layer1_output)
        output = output[1:]
        return output.detach().numpy()
    
    def update(self,x1,x2,output1,output2):
        with torch.no_grad():
            for i in range(self.layer1d.weight.size(0)):
                for j in range(self.layer1d.weight.size(1)):
                    update_fn = self.update_rule[self.learning_rule[i][j]]
                    if self.driving_rule[i,j] == 1:
                        self.layer1d.weight[i][j] = torch.abs(self.layer1d.weight[i][j])
                        self.layer1d.weight[i][j] += update_fn(lr = self.learning_rate[i][j],w = self.layer1d.weight[i][j],x = x1[j],y= output1[i])
                        #self.layer1d.weight[i][j] = self.activation_rule[self.driving_rule1[i,j]](self.layer1d.weight[i][j])
                        if(self.excitatory_rule[i][j]*self.layer1d.weight[i][j] < 0):
                            self.layer1d.weight[i][j] *= self.excitatory_rule[i][j]
                    else:
                        self.layer1m.weight[i][j] = torch.abs(self.layer1m.weight[i][j])
                        self.layer1m.weight[i][j] += update_fn(lr = self.learning_rate[i][j],w = self.layer1m.weight[i][j],x = x2[j],y= output2[i])
                        #self.layer1m.weight[i][j] = self.activation_rule[self.driving_rule1[i,j]](self.layer1m.weight[i][j])
                        if(self.excitatory_rule[i][j]*self.layer1m.weight[i][j] < 0):
                            self.layer1m.weight[i][j] *= self.excitatory_rule[i][j]
                    
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