
import numpy as np
class INetwork:
    def __init__(self,num_sensors,velocity_range,min_velocity,network):
        self.nn = network
        self.alpha = 0.01
        self.velocity_range = velocity_range
        self.tau = 1
        self.A = 1
        self.min_velocity = np.array(min_velocity)
        self.sensor_normalizer = lambda x: self.A + self.A*(self.alpha-1)*(1 - np.exp(-x/self.tau))
        self.scale_velocity_range = 1
    
    def get_input_velocity(self,sensor_input):
        #print("Sensor input is"+str(sensor_input))
        sensor_input = np.array([self.sensor_normalizer(sensor) for sensor in sensor_input],dtype = np.float32)
        #print("Sensor input conv is"+str(sensor_input))
        velocity = self.nn.forward(sensor_input)
        #print("Velocity input conv is"+str(velocity))
        velocity = self.min_velocity + (self.velocity_range/self.scale_velocity_range) * velocity
        velocity = np.round(velocity,2)
        #velocity =(self.velocity_range/self.scale_velocity_range) * velocity
        #print("Velocity input conv is"+str(velocity))
        return velocity[0],velocity[1]



        