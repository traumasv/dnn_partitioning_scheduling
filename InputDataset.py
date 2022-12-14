import sys, copy, math, pickle
import config
import numpy as np
import torch

def get_num_edges():
    return len(config.local_device_info)

def get_total_layers():
    return sum([len(model['layers']) for model in config.service_info])
            
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
    def __init__(self, num_models=3, num_timeslots=get_total_layers(), max_num_partition_slots=10, max_num_layer_in_partition=100, input_size=224*224, len_timeslot=0.05):
        # dtype = ([('device_index', int), ('timeslot_index', int)])
        self.device_layer_schedule = np.full((get_num_edges(), num_timeslots, 2), fill_value=np.nan) # layer based schedule
        self.len_timeslot = 0.05 # each timeslot is 50 ms
        self.num_timeslots = len_timeslot 
        self.input_size = input_size
        self.num_devices = len(config.local_device_info)
        self.service_info = config.service_info[:num_models]
        self.num_services = len(self.service_info)
        # self.layers_in_partitions = [[] for i in range(self.num_services)]
        # self.partition_delay = [[] for i in range(self.num_services)] # comp. + trans. delay
        # self.partition_finish_time = [[] for i in range(self.num_services)] # the finish time of the prev. partition and comp. + trans. delay
        # self.device_partition_schedule = [[] for i in range(self.num_devices)] # partition based schedule (num. of device x num. of device)
        self.layers_in_partitions = np.full((self.num_services, max_num_partition_slots, max_num_layer_in_partition), fill_value=np.nan)
        self.partition_delay = np.full((self.num_services, max_num_partition_slots), fill_value=0.0, dtype=float) # comp. + trans. delay
        self.partition_start_time = np.full((self.num_services, max_num_partition_slots), fill_value=np.nan, dtype=float)
        self.partition_finish_time = np.full((self.num_services, max_num_partition_slots), fill_value=np.nan, dtype=float) # the finish time of the prev. partition and comp. + trans. delay
        self.device_partition_schedule = np.full((self.num_devices, max_num_partition_slots, 2), fill_value=np.nan) # partition based schedule (num. of device x num. of device)
        self.partition_location = np.full((self.num_services, max_num_partition_slots, 2), fill_value=np.nan) # location of each partition [device_index, slot]
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
        devices = self.get_n_devices(model_index=model_index, layer_index=layer_index, num_devices=num_input_partitions) # TODO: this should change as different partitions are assigned on different devices
        weakest_device_index = devices[-1]['device_index']
        comp_delay = (self.service_info[model_index]['layers'][layer_index]['workload_size'] * config.local_device_info[weakest_device_index]['computing_intensity'][model_index])\
                    / (config.local_device_info[weakest_device_index]['computing_frequency'] * num_input_partitions)
        # trans_delay = self.service_info[model_index]['layers'][layer_index]['input_data_size'][0] / (transmission_rate * num_input_partitions) # TODO: take this off, add it to calculate_partition_delay
        return comp_delay
    
    def calculate_partition_delay(self, model_index, partition_index, num_input_partitions = 1):
        partition_delay = 0.0
        # TODO: add transmission delay of the last layer of the partition depending on whether the prev. and next partition is in the same device
        for layer_index in self.layers_in_partitions[model_index][partition_index]:
            if not np.isnan(layer_index):
                partition_delay = partition_delay + self.calculate_layer_delay(model_index, int(layer_index), num_input_partitions=num_input_partitions)
        return partition_delay
    
    def get_finish_time(self):
        last = 0.0
        for model_finish_times in self.partition_finish_time:
            if model_finish_times[~np.isnan(model_finish_times)][-1] > last:
                last = model_finish_times[~np.isnan(model_finish_times)][-1]
        return last

    def get_etas_layer(self, model_index, layer_index): # ETA = earliest time available
        # Input: self.device_layer_schedule
        # Output: an array of all devices [device_index, earliest_timeslot_index] pair
        # in each device, get the min index with nan (if not being used)
        available_device_slot = np.isnan(self.device_layer_schedule)
        last_slot = 0
        # get the slot_index of the prev. layer of the same model
        for device_index, device_schedule in enumerate(self.device_layer_schedule):
            for slot, [scheduled_model, scheduled_layer] in enumerate(device_schedule):
                if scheduled_layer > 0 and scheduled_model == model_index and scheduled_layer == layer_index - 1 and slot > last_slot:
                    last_slot = slot
        # mark all slots that are earlier as not available                    
        for device_index, device_schedule in enumerate(self.device_layer_schedule):
            for slot, [scheduled_model, scheduled_layer] in enumerate(device_schedule):
                if slot <= last_slot:
                    available_device_slot[device_index][slot] = np.array([False, False])
        # create an array with [device_index, earliest_slot_index] pair
        etas = []
        # dtype = ([('device_index', int), ('slot_index', int)])
        for i, x in enumerate(available_device_slot):
            for slot, pair in enumerate(x):
                if pair.any(): # if there are any nan that returned False
                    etas.append(np.array([i, slot]))
                    break
        etas = np.array(etas)
        # sort the device index by the ETA order
        etas = etas[etas[:,1].argsort(axis=0)]
        return etas
    
    def get_eta_for_device(self, model_index, layer_index, device_index):
        # Input: model index (int), layer index (int), device index (int)
        # Output: get etas (array of [device_index, layer_index]) with the correct device index
        etas = self.get_etas_layer(model_index=model_index, layer_index=layer_index)
        device_eta = np.array([[device, slot] for [device, slot] in etas if device == device_index])
        return device_eta

    def get_n_devices(self, model_index, layer_index, num_devices=len(config.local_device_info)):
        # Input: number of devices (int)
        # Output: array of (device index, compute capacity, earliest time available) (int, int, int)
        device_order = get_device_order(model_index=model_index, sort=False)
        etas = self.get_etas_layer(model_index=model_index, layer_index=layer_index)
        dtype = [('device_index', int), ('compute_capacity', int), ('eta', int)]
        device_indices = np.array([tuple(np.append(a,b[1])) for a,b in zip(device_order[device_order[:,0].argsort()], etas[etas[:,0].argsort()])], dtype=dtype)
        # sort with the following priority: 1. ascending earliest eta, then if a tie, 2. negated device compute capacity
        device_indices.sort(order=['eta', 'compute_capacity'])
        # device_indices.sort(order=['compute_capacity', 'eta'])
        return device_indices[:num_devices]

    def calculate_padding_proportion(self, model_index, layer_index, num_partitions): # TODO: turn this from per layer -> per partition
        # Input: model index (int), layer index (int), num_partitions (int)
        # Output: proportion of padding out of the entire partition (float)
        return( ((self.service_info[model_index]['layers'][layer_index]['input_height'] / num_partitions) + self.service_info[model_index]['layers'][layer_index]['padding']) \
                * ((self.service_info[model_index]['layers'][layer_index]['input_width'] / num_partitions) + self.service_info[model_index]['layers'][layer_index]['padding']) \
                - (((self.service_info[model_index]['layers'][layer_index]['input_height'] / num_partitions)) \
                * ((self.service_info[model_index]['layers'][layer_index]['input_width'] / num_partitions) )) ) \
                / ((self.service_info[model_index]['layers'][layer_index]['input_height'] / num_partitions) * (self.service_info[model_index]['layers'][layer_index]['input_width'] / num_partitions))

    def get_num_input_partitions(self, model_index, partition_index, padding_threshold=0.50): # TODO: turn this from per layer -> per partition
        # Input: input_size (int)
        # Output: num_partitions (int), input_partition_size (int)
        temp = 1
        num_input_partitions = 1
        t_num_partitions = 10000000000
        input_partition_size = int(self.service_info[model_index]['layers'][partition_index]['input_data_size'][0] / temp)
        t_temp = self.calculate_partition_delay(
                                            model_index=model_index, 
                                            partition_index=partition_index, 
                                            num_input_partitions=temp
                                            )
        while t_num_partitions > t_temp \
                and self.calculate_padding_proportion(model_index=model_index, partition_index=partition_index, num_partitions=temp) < padding_threshold \
                and num_partitions < get_num_edges():
            num_partitions = temp
            t_num_partitions = t_temp
            temp = temp+1
            t_temp = self.calculate_layer_delay(
                                                model_index=model_index,
                                                partition_index=partition_index,
                                                num_input_partitions=temp
                                                )
        return num_partitions, int(self.service_info[model_index]['layers'][partition_index]['input_data_size'][0] / num_partitions)

##########################################################################################################################################################################################

    def calculate_waiting_delay(self, device_index, slot_index, model_index, partition_index):
        if partition_index == 0:
            return 0.0
        if partition_index > 0:
            last_finish_time = self.partition_finish_time[model_index][partition_index-1] 
        model_i, partition_i = self.device_partition_schedule[device_index][slot_index]
        waiting_delay = self.partition_finish_time[model_i][partition_i] - last_finish_time
        if waiting_delay < 0:
            return 0.0
        return waiting_delay

    def get_earliest_slot_in_device(self, device_index):
        # Input: device index (int)
        # Output: slot index with the first np.nan (int)
        return np.argmax(np.isnan(self.device_partition_schedule[device_index]).any(axis=1))

    def get_crevice_slots(self, model_index, partition_index):
        # TODO: get time gaps between partitions and reschedule
        self.partition_start_time[model_index][partition_index]
        return

    def get_partition_finish_time(self, model_index, partition_index, device_index, num_partitions=1):
        # Input: self.device_layer_schedule (array of device x timeslot x [model_index, layer_index]), partition_index
        # Output: finishing time for current partition (float)
        slot_index = self.get_earliest_slot_in_device(device_index)
        print('slot_index: ', slot_index)
        self.partition_start_time[model_index][partition_index]
        prev_model_index, prev_partition_index = self.device_partition_schedule[device_index][slot_index-1] # get the model and partition that is right before the earliest available slot
        if slot_index == 0:
            if partition_index == 0:
                return self.calculate_partition_delay(model_index, partition_index)
            elif partition_index > 0:
                return self.partition_finish_time[model_index][partition_index-1] + self.calculate_partition_delay(model_index, partition_index) 
        elif slot_index > 0:
            prev_model_index, prev_partition_index = int(prev_model_index), int(prev_partition_index)
            if partition_index == 0:
                return self.partition_finish_time[prev_model_index][prev_partition_index] + self.calculate_partition_delay(model_index, partition_index) 
            elif partition_index > 0:
                if self.partition_finish_time[prev_model_index][prev_partition_index] > self.partition_finish_time[model_index][partition_index-1]:
                    return self.partition_finish_time[prev_model_index][prev_partition_index] + self.calculate_partition_delay(model_index, partition_index) 
                elif self.partition_finish_time[prev_model_index][prev_partition_index] <= self.partition_finish_time[model_index][partition_index-1]:
                    return self.partition_finish_time[model_index][partition_index-1] + self.calculate_partition_delay(model_index, partition_index) 
        # TODO: get the last finish time of the input partitions if the num. of partitions is > 1

    def reschedule_partition(self, model_index, partition_index):
        # Input: model index (int), layer index (int), number of partitions (int)
        # Output: a device and queue index pair (list) to relocate the partition
        # start with the current assignment as the standard
        current_device, current_slot = self.partition_location[model_index][partition_index]
        current_device, current_slot = int(current_device), int(current_slot)
        earliest_device, earliest_slot = int(current_device), int(current_slot)
        earliest_finish_time = self.partition_finish_time[model_index][partition_index]
        # search every device and get the earliest slot in each device
        for device_index, _ in enumerate(config.local_device_info):
            device_slot = self.get_earliest_slot_in_device(device_index)
            new_finish_time = self.get_partition_finish_time(model_index, partition_index, device_index)
            if new_finish_time < earliest_finish_time:
                earliest_device, earliest_slot = device_index, int(device_slot)
                earliest_finish_time = new_finish_time
        if earliest_device == current_device and earliest_slot >= current_slot:
            return
        # TODO: update the start time
        self.device_partition_schedule[earliest_device][earliest_slot] = self.device_partition_schedule[current_device][current_slot]
        self.device_partition_schedule[current_device][current_slot] = np.array([np.nan, np. nan])
        self.partition_location[model_index][partition_index] = np.array([earliest_device, earliest_slot])
        self.partition_delay[model_index][partition_index] = self.calculate_partition_delay(model_index, partition_index)
        self.partition_finish_time[model_index][partition_index] = earliest_finish_time
        print('earliest_finish_time after : ', earliest_finish_time)
    
    def reschedule_partitions(self):
        for model_index, partitions in enumerate(self.partition_location):
            for partition_index, [device_index, device_slot_index] in enumerate(self.partition_location[model_index]):
                if not np.isnan(device_index):
                    print('model_index: ',model_index)
                    print('partition_index: ',partition_index)
                    self.reschedule_partition(model_index, partition_index)
                    print('self.partition_location: ',self.partition_location)
                    print('self.device_partition_schedule: ', self.device_partition_schedule)
                    print('self.partition_delay: ', self.partition_delay)
                    print('self.partition_finish_time: ',self.partition_finish_time)
                    input()
        return self.get_finish_time()
                
##########################################################################################################################################################################################
    
    def create_init_model_partition(self, device_usage_threshold=0.400): # create init partitions based on the time usage limit
        for model_index, models in enumerate(self.service_info):
            print('Model %d'%model_index)
            layer_index = 0
            partition_index = 0
            partition_start_time = 0.0
            for device_index, device_schedule in enumerate(self.device_layer_schedule):
                layer_index_in_partition = 0
                # if self.service_info[model_index]['layers'][layer_index]['layer_type'] == 'fc':
                #     optimal_num_partitions = 1
                # else:
                #     optimal_num_partitions, input_partition_size = self.get_num_input_partitions(model_index, layer_index)
                optimal_num_partitions = 1
                # if the layer is less than the threshold add layer to partition
                print('Layer %d'%layer_index)
                # print('Partition delay w/o layer = ',
                # self.partition_delay[model_index][partition_index] 
                # )
                # print('Layer delay = ',
                # self.calculate_layer_delay(
                #                         model_index=model_index,
                #                         layer_index=layer_index,
                #                         num_input_partitions=optimal_num_partitions
                #                         )    
                # )
                # print('Partition delay w/ layer = ',
                # self.partition_delay[model_index][partition_index]  \
                #                             + \
                # self.calculate_layer_delay(
                #                         model_index=model_index,
                #                         layer_index=layer_index,
                #                         num_input_partitions=optimal_num_partitions
                #                         )
                # )                
                while (self.partition_delay[model_index][partition_index]  \
                                                + \
                    self.calculate_layer_delay(
                                            model_index=model_index,
                                            layer_index=layer_index,
                                            num_input_partitions=optimal_num_partitions
                                            ) < device_usage_threshold) \
                                            and (layer_index < len(self.service_info[model_index]['layers']) - 1):
                    eta = self.get_eta_for_device(model_index, layer_index, device_index)[0][1] # first device on the list, and available timeslot [device_index, timeslot_index]
                    self.layers_in_partitions[model_index][partition_index][layer_index_in_partition] = layer_index 
                    layer_index_in_partition += 1
                    self.partition_delay[model_index][partition_index] = \
                        self.partition_delay[model_index][partition_index] \
                                                    + \
                        self.calculate_layer_delay(
                                                model_index=model_index,
                                                layer_index=layer_index,
                                                num_input_partitions=optimal_num_partitions
                                                )
                    self.device_layer_schedule[device_index][eta] = [model_index, layer_index]
                    layer_index += 1
                if not np.isnan(self.layers_in_partitions[model_index][partition_index]).all(): # check if it's a partition with layers inside
                    # TODO: finish input partitioning for the created layer group partitions
                    partition_index_in_device = self.get_earliest_slot_in_device(device_index)
                    self.partition_location[model_index][partition_index] = [device_index, partition_index_in_device]
                    self.partition_finish_time[model_index][partition_index] = self.get_partition_finish_time(model_index, partition_index, device_index) # get_partition_finish_time() only works for when initially creating the partition
                    self.device_partition_schedule[device_index][partition_index_in_device] = [model_index, partition_index]
                    # set start time
                    if partition_index_in_device == 0:
                        if device_index == 0:
                            self.partition_start_time[model_index][partition_index] = 0.0
                        else:
                            prev_device_earliest_slot = self.get_earliest_slot_in_device(device_index-1) - 1
                            prev_model, prev_partition = self.device_partition_schedule[device_index-1][prev_device_earliest_slot]
                            prev_model, prev_partition = int(prev_model), int(prev_partition)
                            self.partition_start_time[model_index][partition_index] = self.partition_finish_time[prev_model][prev_partition]
                    else:
                        prev_model, prev_partition = self.device_partition_schedule[device_index][partition_index_in_device-1]
                        prev_model, prev_partition = int(prev_model), int(prev_partition)
                        self.partition_start_time[model_index][partition_index] = self.partition_finish_time[prev_model][prev_partition]
                    partition_index += 1
                    print('self.partition_delay: ', self.partition_delay)
                    print('self.partition_start_time: ',self.partition_start_time)
                    print('self.partition_finish_time: ',self.partition_finish_time)
                    print('self.partition_location: ', self.partition_location)
                    # input() 
        return self.get_finish_time()
ds = InputDataSet()
print('Finish time before re-scheduling:',ds.create_init_model_partition())
input()
print('Finish time after rescheduling: ',ds.reschedule_partitions())