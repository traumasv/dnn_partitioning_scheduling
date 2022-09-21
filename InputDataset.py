
from pyexpat import model
import sys, copy, math
import config
import numpy as np
import torch
# np.set_printoptions(threshold=sys.maxsize)

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
    # def __init__(self, num_timeslots=get_total_layers(), input_size=224*224):
    def __init__(self, num_timeslots=15, input_size=224*224, len_timeslot=0.05):
        # dtype = ([('device_index', int), ('timeslot_index', int)])
        self.schedule = np.full((get_num_edges(), num_timeslots, 2), fill_value=np.nan)
        self.len_timeslot = 0.05 # each timeslot is 50 ms
        self.num_timeslots = len_timeslot 
        self.num_services = len(config.service_info)
        self.input_size = input_size
        # self.service_info = get_service_info()
        self.service_info = config.service_info
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
        # check which device is the slowest to compute 
        devices = self.get_n_devices(model_index=model_index, layer_index=layer_index, num_devices=num_input_partitions)
        weakest_device_index = devices[-1]['device_index']
        comp_delay = (self.service_info[model_index]['layers'][layer_index]['workload_size'] * config.local_device_info[weakest_device_index]['computing_intensity'][model_index])\
                    / (config.local_device_info[weakest_device_index]['computing_frequency'] * num_input_partitions)
        trans_delay = self.service_info[model_index]['layers'][layer_index]['input_data_size'][0] / (transmission_rate * num_input_partitions) # input data size is a single elem. arr.
        # if num_input_partitions == 1:
        #     return comp_delay
        return comp_delay + trans_delay
    
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
        return last_timeslot_used

    # An eta is dependent on the eta of other devices, this method IS NOT CORRECT
    # def get_device_eta(self, device_index, model_index, layer_index): 
    #     # Input: device_index (int), model_index (int), layer_index (int), self.schedule[device_index] (single arr. of [timeslot])
    #     # Output: device_eta (int), or None when there are no timeslots available
    #     available_timeslot = np.isnan(self.schedule[device_index])
    #     for timeslot, [scheduled_model, scheduled_layer] in enumerate(self.schedule[device_index]):
    #         if scheduled_model == model_index and scheduled_layer < layer_index:
    #             available_timeslot[timeslot] = np.array([False, False])
    #     device_eta = np.argwhere(available_timeslot.any(axis=1) == True).flatten()
    #     if len(device_eta) > 0:
    #         return device_eta[0] # take the first available timeslot in the device
    #     return None

    def get_etas(self, model_index, layer_index): # ETA = earliest time available
        # Input: self.schedule
        # Output: an array of all devices [device_index, earliest_timeslot_index] pair
        # in each device, get the min index with nan (if not being used)
        available_device_timeslot = np.isnan(self.schedule)
        # (check for the last layer in the model then get its timeslot, make sure all etas returned in each device is not earlier than the timeslot of the current layer)
        for device_index, device_schedule in enumerate(self.schedule):
            for timeslot, [scheduled_model, scheduled_layer] in enumerate(device_schedule):
                if scheduled_model == model_index and scheduled_layer < layer_index: # TODO: add a check for all timeslots that are earlier than the finishing timeslot of the last layer
                    available_device_timeslot[device_index] = [False, False]
        # create an array with [device_index, earliest_timeslot_index] pair
        etas = []
        for i, x in enumerate(available_device_timeslot):
            for timeslot, pair in enumerate(x):
                if pair.any(): # if there are any nan that returned False
                    etas.append(np.array([i, timeslot]))
                    break
        etas = np.array(etas)
        # sort the device index by the ETA order
        etas = etas[etas[:,1].argsort(axis=0)]
        return etas

    def get_n_devices(self, model_index, layer_index, num_devices):
        # Input: number of devices (int), number of timeslots (int)
        # Output: array of (device index, compute capacity, earliest time available) (int, int, int)
        # sort with the following priority: 1. ascending earliest eta, then if a tie, 2. negated device compute capacity
        device_order = get_device_order(model_index=model_index, sort=False)
        etas = self.get_etas(model_index=model_index, layer_index=layer_index)
        dtype = [('device_index', int), ('compute_capacity', int), ('eta', int)]
        device_indices = np.array([tuple(np.append(a,b[1])) for a,b in zip(device_order[device_order[:,0].argsort()], etas[etas[:,0].argsort()])], dtype=dtype)
        device_indices.sort(order=['eta', 'compute_capacity'])
        # device_indices.sort(order=['compute_capacity', 'eta'])
        # TODO: handle case where all devices are full
        return device_indices[:num_devices]

    # TODO: is there a difference between create input partitions and schedule? if NO, combine into create input partitions
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
            < \
            self.calculate_layer_delay(
                                        model_index=model_index, 
                                        layer_index=layer_index, 
                                        num_partitions=prev
                                        ):
                self.schedule[device_index, timeslot_index] = [model_index, layer_index] 
                break

    def create_input_partitions(self, model_index, layer_index, padding_threshold=0.50): # assigns duplicate layers to devices with different sized inputs
        # Input: input_size (int)
        # Output: num_partitions (int), partition_size (int)
        temp = 1
        num_partitions = 1
        t_num_partitions = 10000000000
        partition_size = int(self.service_info[model_index]['layers'][layer_index]['input_data_size'][0] / temp)
        t_temp = self.calculate_layer_delay(
                                            model_index=model_index, 
                                            layer_index=layer_index, 
                                            num_partitions=temp
                                            )
        while t_num_partitions > t_temp \
                and self.calculate_proportion(model_index=model_index, layer_index=layer_index, num_partitions=temp) < padding_threshold \
                and num_partitions < get_num_edges():
            # input()
            num_partitions = temp
            t_num_partitions = t_temp
            temp = int(math.pow((int(math.sqrt(temp)) + 1), 2))
            t_temp = self.calculate_layer_delay(
                                                model_index=model_index,
                                                layer_index=layer_index,
                                                num_partitions=temp
                                                )
        return num_partitions, int(self.service_info[model_index]['layers'][layer_index]['input_data_size'][0] / num_partitions)

    def create_model_partitions(self, delay_threshold=0.200): # method for layer-wise re-scheduling
        # Input:  partition_delay_threshold (float), delay_threshold
        # Output: finish_time
        for model_index, models in enumerate(self.service_info):
            for layer_index, layer in enumerate(models['layers']):
                # TODO: add input partitioning
                # num_partitions = 1
                # if layer['layer_type'] == 'cnn':
                #     num_partitions, _ = self.create_input_partitions(model_index=model_index, layer_index=layer_index)
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
                        print(self.schedule)    
                        if model_index == get_num_models()-1 and layer_index == get_num_layers(model_index=model_index)-1:
                            return self.get_finish_time()
                        continue # continue onto the next layer
    
    def create_init_model_partition(self, device_usage_threshold=0.200): # create init partitions based on the time usage limit
        device_usage_threshold_timeslot = int(device_usage_threshold / self.len_timeslot)
        for model_index, models in enumerate(self.service_info):
            print('Model %d'%model_index)
            layer_index = 0
            for device_index, device_schedule in enumerate(self.schedule):
                while self.calculate_layer_delay(
                                            model_index=model_index,
                                            layer_index=layer_index,
                                            num_input_partitions=1
                                            ) < device_usage_threshold \
                                            and layer_index < device_usage_threshold_timeslot +  device_index * device_usage_threshold_timeslot \
                                            and layer_index < len(self.service_info[model_index]['layers'][layer_index]) - 1:
                    # if the layer delay is less than the delay threshold continue assigning another timeslot
                    print('Layer %d'%layer_index)
                    timeslot_taken = math.ceil(self.calculate_layer_delay(
                                            model_index=model_index,
                                            layer_index=layer_index,
                                            num_input_partitions=1
                                            ) / self.len_timeslot)
                    if layer_index > 0:
                        self.schedule[device_index, self.get_etas(model_index, layer_index-1)[device_index]] = [model_index, layer_index]
                    else:
                        self.schedule[device_index, self.get_etas(model_index, layer_index)[device_index]] = [model_index, layer_index]
                    # check if the num. of timeslot is less than 
                    layer_index += 1
                    print(self.schedule)
                    input()

ds = InputDataSet()
# print(ds.get_etas())
# print(get_device_order(0))
# print(ds.calculate_layer_delay(0,0, device_indices=[0,1,2,3], num_partitions=4))
# ds.schedule_input_partition(model_index=0, layer_index=0, num_partitions=4)
# print(ds.create_input_partitions(0,0))
# print('Total Timeslots Taken: %d'%ds.create_model_partitions())
# print('Total Timeslots Taken: ', ds.create_model_partitions())
print(ds.create_init_model_partition())