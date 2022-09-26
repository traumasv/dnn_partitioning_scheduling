
from pyexpat import model
import sys, copy, math
import config
import numpy as np
import torch

def get_num_edges():
    return len(config.local_device_info)

def get_total_layers():
    return sum([len(model['layers']) for model in config.service_info])

def get_num_models():
    return len(config.service_info)

def get_num_layers(model_index):
    return len(config.service_info[model_index]['layers'])
            
def get_device_order(model_index, sort=True):
    # Input: model index (int)
    # Output: ordered list of [device index, computing capacity] in dec. computing capacity -(computing frequency / computing intensity) order
    device_order = np.array(
                            [[device_index, -(device_info['computing_frequency'] / device_info['computing_intensity'][model_index])] for device_index, device_info in enumerate(config.local_device_info)], 
                            dtype=int
                            )
    if sort == True:
        device_order = np.array(
                                device_order[device_order[:,1].argsort(axis=0)], 
                                dtype=int
                                )
    return device_order

class InputDataSet():
    def __init__(self, num_timeslots=get_total_layers(), input_size=224*224, len_timeslot=0.05):
        # dtype = ([('device_index', int), ('timeslot_index', int)])
        self.schedule = np.full((get_num_edges(), num_timeslots, 2), fill_value=np.nan) # layer based schedule
        self.timeslot_schedule = np.full((get_num_edges(), num_timeslots, 2), fill_value=np.nan)
        self.len_timeslot = 0.05 # each timeslot is 50 ms
        self.num_timeslots = len_timeslot 
        self.input_size = input_size
        # self.service_info = get_service_info() #TODO: finish the method to select which services to deploy from config file
        self.service_info = config.service_info
        self.num_services = len(self.service_info)
        self.partitions = [[] for i in range(self.num_services)]
        self.partition_delay = [[] for i in range(self.num_services)]
        for dnn in self.service_info:
            for layer_info in dnn['layers']:
                if layer_info['layer_type'] == 'cnn':
                    layer_info['workload_size'] *= layer_info['output_height'] * layer_info['output_width'] * layer_info['output_channel'] * layer_info['input_channel'] * layer_info['kernel'] * layer_info['kernel']
                    layer_info['memory'] = layer_info['kernel'] * layer_info['kernel'] * layer_info['input_channel'] * layer_info['output_channel'] * 4 + layer_info['output_channel'] * 4
                    layer_info['memory'] += layer_info['input_height'] * layer_info['input_width'] * layer_info['input_channel'] * 4
                elif layer_info['layer_type'] == 'maxpool' or layer_info['layer_type'] == 'avgpool':
                    layer_info['workload_size'] *= layer_info['output_height'] * layer_info['output_width'] * layer_info['output_channel'] * layer_info['kernel'] * layer_info['kernel']
                    layer_info['memory'] = layer_info['input_height'] * layer_info['input_width'] * layer_info['input_channel'] * 4
                elif layer_info['layer_type'] == 'fc':
                    layer_info['workload_size'] *= layer_info['output_height'] * layer_info['output_width'] * layer_info['output_channel'] * (2 * layer_info['input_height'] * layer_info['input_width'] * layer_info['input_channel'] - 1)
                    layer_info['memory'] = layer_info['input_height'] * layer_info['input_width'] * layer_info['input_channel'] * layer_info['output_channel'] * 4 + layer_info['output_channel'] * 4
                    layer_info['memory'] += layer_info['input_height'] * layer_info['input_width'] * layer_info['input_channel'] * 4
                '''
                Overhead
                CL = output_height * output width * output_channel * input_channel * kernel * kernel
                '''
    def calculate_layer_delay(self, model_index, layer_index, transmission_rate=10*(10**6), num_input_partitions=1): # Model index: 0 - AlexNet, 1 - GoogLeNet, 2 - ResNet-50 
        # Input: model index (int), layer index (int), list of device index/indices [int,..], number of partitions (int)
        # Output: total delay (float)
        # get the slowest completion time out of the partitions
        devices = self.get_n_devices(model_index=model_index, layer_index=layer_index, num_devices=num_input_partitions)
        weakest_device_index = devices[-1]['device_index'] # This might not be true
        comp_delay = (self.service_info[model_index]['layers'][layer_index]['workload_size'] * config.local_device_info[weakest_device_index]['computing_intensity'][model_index])\
                    / (config.local_device_info[weakest_device_index]['computing_frequency'] * num_input_partitions)
        trans_delay = self.service_info[model_index]['layers'][layer_index]['input_data_size'][0] / (transmission_rate * num_input_partitions) # input data size is a single elem. arr.
        return comp_delay + trans_delay
    
    def calculate_partition_delay(self, model_index, partition_index, num_input_partitions = 1):
        partition_delay = 0.0
        for layer_index in self.partitions[model_index][partition_index]:
            partition_delay = partition_delay + self.calculate_layer_delay(model_index, layer_index, num_input_partitions=num_input_partitions)
        return partition_delay
    
    def calculate_proportion(self, model_index, layer_index, num_partitions):
        # Input: model index (int), layer index (int), num_partitions (int)
        # Output: proportion of padding out of the entire partition (float)
        return( ((self.service_info[model_index]['layers'][layer_index]['input_height'] / num_partitions) + self.service_info[model_index]['layers'][layer_index]['padding']) \
                * ((self.service_info[model_index]['layers'][layer_index]['input_width'] / num_partitions) + self.service_info[model_index]['layers'][layer_index]['padding']) \
                - (((self.service_info[model_index]['layers'][layer_index]['input_height'] / num_partitions)) \
                * ((self.service_info[model_index]['layers'][layer_index]['input_width'] / num_partitions) )) ) \
                / ((self.service_info[model_index]['layers'][layer_index]['input_height'] / num_partitions) * (self.service_info[model_index]['layers'][layer_index]['input_width'] / num_partitions))

    def get_finish_time(self):
        # Input: self.schedule (array of device x timeslot x [model_index, layer_index])
        # Output: finishing timeslot for all services (int)
        available_timeslot_bool_arr = np.isnan(self.schedule).any(axis=2)
        available_timeslot_index_arr = np.argwhere(available_timeslot_bool_arr == True)
        last_timeslot_used = 0
        for [device_index, timeslot] in available_timeslot_index_arr:
            if timeslot > last_timeslot_used:
                last_timeslot_used = timeslot
        print(available_timeslot_index_arr)
        return last_timeslot_used

    def get_etas(self, model_index, layer_index): # ETA = earliest time available
        # Input: self.schedule
        # Output: an array of all devices [device_index, earliest_timeslot_index] pair
        # in each device, get the min index with nan (if not being used)
        available_device_timeslot = np.isnan(self.schedule)
        last_timeslot = 0
        # get the timeslot_index of the prev. layer of the same model
        for device_index, device_schedule in enumerate(self.schedule):
            for timeslot, [scheduled_model, scheduled_layer] in enumerate(device_schedule):
                if scheduled_layer > 0 and scheduled_model == model_index and scheduled_layer == layer_index - 1 and timeslot > last_timeslot:
                    last_timeslot = timeslot
        # mark all timeslots that are earlier as not available                    
        for device_index, device_schedule in enumerate(self.schedule):
            for timeslot, [scheduled_model, scheduled_layer] in enumerate(device_schedule):
                if timeslot <= last_timeslot:
                    available_device_timeslot[device_index][timeslot] = np.array([False, False])
        # create an array with [device_index, earliest_timeslot_index] pair
        etas = []
        # dtype = ([('device_index', int), ('timeslot_index', int)])
        for i, x in enumerate(available_device_timeslot):
            for timeslot, pair in enumerate(x):
                if pair.any(): # if there are any nan that returned False
                    etas.append(np.array([i, timeslot]))
                    break
        etas = np.array(etas)
        # sort the device index by the ETA order
        etas = etas[etas[:,1].argsort(axis=0)]
        return etas
    
    def get_eta_device(self, model_index, layer_index, device_index):
        etas = self.get_etas(model_index=model_index, layer_index=layer_index)
        device_eta = np.array([[device, timeslot] for [device, timeslot] in etas if device == device_index])
        return device_eta

    def get_n_devices(self, model_index, layer_index, num_devices=len(config.local_device_info)):
        # Input: number of devices (int), number of timeslots (int)
        # Output: array of (device index, compute capacity, earliest time available) (int, int, int)
        device_order = get_device_order(model_index=model_index, sort=False)
        etas = self.get_etas(model_index=model_index, layer_index=layer_index)
        dtype = [('device_index', int), ('compute_capacity', int), ('eta', int)]
        device_indices = np.array([tuple(np.append(a,b[1])) for a,b in zip(device_order[device_order[:,0].argsort()], etas[etas[:,0].argsort()])], dtype=dtype)
        # sort with the following priority: 1. ascending earliest eta, then if a tie, 2. negated device compute capacity
        device_indices.sort(order=['eta', 'compute_capacity'])
        # device_indices.sort(order=['compute_capacity', 'eta'])
        return device_indices[:num_devices]

    # TODO: turn into re-schedule partitions
    def schedule_input_partition(self, model_index, layer_index, num_partitions, prev, timeslot_index):
        # Input: model index (int), layer index (int), number of partitions (int)
        # Output: self.schedule (array with size: num_device x num_timeslot x 2)
        devices = self.get_n_devices(model_index=model_index, layer_index=layer_index, num_devices=num_partitions)
        for device_index in self.get_n_devices(model_index=model_index, layer_index=layer_index, num_devices=num_partitions):
            if self.service_info[model_index]['layers'][layer_index]['memory'] / num_partitions < config.local_device_info[device_index]['memory'] \
            and self.calculate_layer_delay(
                                            model_index=model_index, 
                                            layer_index=layer_index, 
                                            num_partitions=num_partitions 
                                            ) \
            <   self.calculate_layer_delay(
                                        model_index=model_index, 
                                        layer_index=layer_index, 
                                        num_partitions=prev
                                        ):
                self.schedule[device_index, timeslot_index] = [model_index, layer_index] 
                break

    def get_num_input_partitions(self, model_index, layer_index, padding_threshold=0.50): # calculates the optimal input partition
        # Input: input_size (int)
        # Output: num_partitions (int), input_partition_size (int)
        temp = 1
        num_partitions = 1
        t_num_partitions = 10000000000
        partition_size = int(self.service_info[model_index]['layers'][layer_index]['input_data_size'][0] / temp)
        t_temp = self.calculate_layer_delay(
                                            model_index=model_index, 
                                            layer_index=layer_index, 
                                            num_input_partitions=temp
                                            )
        while t_num_partitions > t_temp \
                and self.calculate_proportion(model_index=model_index, layer_index=layer_index, num_partitions=temp) < padding_threshold \
                and num_partitions < get_num_edges():
            num_partitions = temp
            t_num_partitions = t_temp
            temp = temp+1
            t_temp = self.calculate_layer_delay(
                                                model_index=model_index,
                                                layer_index=layer_index,
                                                num_input_partitions=temp
                                                )
        return num_partitions, int(self.service_info[model_index]['layers'][layer_index]['input_data_size'][0] / num_partitions)

    def create_model_partitions(self, delay_threshold=0.200): # method for layer-wise re-scheduling
        # Input:  partition_delay_threshold (float), delay_threshold (float)
        # Output: finish_time (float)
        for model_index, models in enumerate(self.service_info):
            for layer_index, layer in enumerate(models['layers']):
                while self.calculate_layer_delay(
                                            model_index=model_index, 
                                            layer_index=layer_index, 
                                            num_input_partitions=1
                                            ) < delay_threshold:
                    # if the layer delay is less than the delay threshold continue assigning another timeslot OR the layer delay 
                    print('New Assignment')
                    input()
                    for (device_index, compute_capacity, eta) in self.get_n_devices(model_index=model_index, layer_index=layer_index, num_devices=1):
                        self.schedule[device_index, eta] = [model_index, layer_index]
                        if model_index == get_num_models()-1 and layer_index == get_num_layers(model_index=model_index)-1:
                            return self.get_finish_time()
                        continue # continue onto the next layer
    
    def create_init_model_partition(self, device_usage_threshold=0.400): # create init partitions based on the time usage limit
        device_usage_threshold_timeslot = int(device_usage_threshold / self.len_timeslot)
        for model_index, models in enumerate(self.service_info):
            print('Model %d'%model_index)
            layer_index = 0
            partition_index = 0
            for device_index, device_schedule in enumerate(self.schedule):
                # begin a partition
                self.partitions[model_index].append([])  
                self.partition_delay[model_index].append(0.0) 
                if config.service_info[model_index]['layers'][layer_index]['layer_type'] == 'fc':
                    optimal_num_partitions = 1
                else:
                    optimal_num_partitions, input_partition_size = self.get_num_input_partitions(model_index, layer_index)
                # if the layer is less than the threshold add layer to partition
                print('Layer %d'%layer_index)
                print('Partition delay w/o layer = ',
                self.partition_delay[model_index][partition_index] 
                )
                print('Layer delay = ',
                self.calculate_layer_delay(
                                        model_index=model_index,
                                        layer_index=layer_index,
                                        num_input_partitions=optimal_num_partitions
                                        )    
                )
                print('Partition delay w/ layer = ',
                self.partition_delay[model_index][partition_index]  \
                                            + \
                self.calculate_layer_delay(
                                        model_index=model_index,
                                        layer_index=layer_index,
                                        num_input_partitions=optimal_num_partitions
                                        )
                )                
                while (self.partition_delay[model_index][partition_index]  \
                                                + \
                    self.calculate_layer_delay(
                                            model_index=model_index,
                                            layer_index=layer_index,
                                            num_input_partitions=optimal_num_partitions
                                            ) < device_usage_threshold) \
                                            and (layer_index < len(self.service_info[model_index]['layers']) - 1):
                    eta = self.get_eta_device(model_index, layer_index, device_index)[0][1] #first device on the list, and available timeslot [device_index, timeslot_index]
                    # input()
                    self.partitions[model_index][partition_index].append(layer_index)
                    self.partition_delay[model_index][partition_index] = \
                        self.partition_delay[model_index][partition_index] \
                                                    + \
                        self.calculate_layer_delay(
                                                model_index=model_index,
                                                layer_index=layer_index,
                                                num_input_partitions=optimal_num_partitions
                                                )
                    # TODO: add the layer to the schedule with adjusted num. of timeslots taken by the layer(?)
                    # timeslot_taken = math.ceil(self.calculate_partition_delay(
                    #                         model_index=model_index,
                    #                         partition_index=partition_index,
                    #                         num_input_partitions=1
                    #                         ) / self.len_timeslot)                           
                    # for t in range(timeslot_taken):
                    print('self.schedule[device_index][eta]: ',self.schedule[device_index][eta])
                    self.schedule[device_index][eta] = [model_index, layer_index]
                    layer_index += 1
                    print('schedule: ',self.schedule)
                    print('partitions: ',self.partitions)
                    print('partition_delay: ',self.partition_delay)
                    # input()
                partition_index += 1
        return self.get_finish_time()
ds = InputDataSet()
# print(ds.get_etas(0,0))
# print(get_device_order(0))
# print(ds.calculate_layer_delay(0,0, device_indices=[0,1,2,3], num_partitions=4))
# ds.schedule_input_partition(model_index=0, layer_index=0, num_partitions=4)
# print(ds.get_num_input_partitions(0,0))
# print('Total Timeslots Taken: %d'%ds.create_model_partitions())
# print('Total Timeslots Taken: ', ds.create_model_partitions())
print('Finish time before re-scheduling:',ds.create_init_model_partition())
# print('total layers: ',get_total_layers())