import sys, copy, math
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
        # get the slowest completion time out of the partitions
        devices = self.get_n_devices(model_index=model_index, layer_index=layer_index, num_devices=num_input_partitions) # FIXME this should change as different partitions are assigned on different devices
        weakest_device_index = devices[-1]['device_index']
        comp_delay = (self.service_info[model_index]['layers'][layer_index]['workload_size'] * config.local_device_info[weakest_device_index]['computing_intensity'][model_index])\
                    / (config.local_device_info[weakest_device_index]['computing_frequency'] * num_input_partitions)
        trans_delay = self.service_info[model_index]['layers'][layer_index]['input_data_size'][0] / (transmission_rate * num_input_partitions) # input data size is a single elem. arr.
        return comp_delay + trans_delay
    
    def calculate_partition_delay(self, model_index, partition_index, num_input_partitions = 1):
        partition_delay = 0.0
        for layer_index in self.layers_in_partitions[model_index][partition_index]:
            if not np.isnan(layer_index):
                partition_delay = partition_delay + self.calculate_layer_delay(model_index, layer_index, num_input_partitions=num_input_partitions)
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
        available_device_timeslot = np.isnan(self.device_layer_schedule)
        last_timeslot = 0
        # get the timeslot_index of the prev. layer of the same model
        for device_index, device_schedule in enumerate(self.device_layer_schedule):
            for timeslot, [scheduled_model, scheduled_layer] in enumerate(device_schedule):
                if scheduled_layer > 0 and scheduled_model == model_index and scheduled_layer == layer_index - 1 and timeslot > last_timeslot:
                    last_timeslot = timeslot
        # mark all timeslots that are earlier as not available                    
        for device_index, device_schedule in enumerate(self.device_layer_schedule):
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
    
    def get_eta_for_device(self, model_index, layer_index, device_index):
        # Input: model index (int), layer index (int), device index (int)
        # Output: get etas (array of [device_index, layer_index]) with the correct device index
        etas = self.get_etas_layer(model_index=model_index, layer_index=layer_index)
        device_eta = np.array([[device, timeslot] for [device, timeslot] in etas if device == device_index])
        return device_eta

    def get_n_devices(self, model_index, layer_index, num_devices=len(config.local_device_info)):
        # Input: number of devices (int), number of timeslots (int)
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

    def get_num_input_partitions(self, model_index, layer_index, padding_threshold=0.50): # TODO: turn this from per layer -> per partition
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
                and self.calculate_padding_proportion(model_index=model_index, layer_index=layer_index, num_partitions=temp) < padding_threshold \
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

##########################################################################################################################################################################################
    def get_prev_slot_device_queue(self, model_index, partition_index, device_index): 
        for device_queue_i, [model_i, partition_i] in enumerate(self.device_partition_schedule[device_index]):
            if np.isnan(model_i):
                return None
            if model_i == model_index and partition_i == partition_index:
                if device_queue_i < 1:
                    return None
                return device_queue_i - 1

    def get_etas_partition_device(model_index, partition_index, device_index):
        return

    def get_partition_finish_time(self, model_index, partition_index, device_index, num_partitions=1):
        # Input: self.device_layer_schedule (array of device x timeslot x [model_index, layer_index]), partition_index
        # Output: finishing time for current partition (float)
        prev_i = self.get_prev_slot_device_queue(model_index, partition_index, device_index)
        if prev_i == None:
            return self.partition_delay[model_index][partition_index]
        # TODO: add waiting delay = earliest eta available - finish time of the last partition in the same model
        waiting_delay = self.partition_finish_time[model_index][partition_index-1]
        if partition_index == 0:
            [model_i, partition_i] = self.partition_finish_time[device_index][prev_i]
            return self.partition_finish_time[model_i][partition_i] + self.partition_delay[model_index][partition_index]
        if num_partitions == 1:
            partition_finish_time = self.partition_finish_time[model_index][partition_index-1] + waiting_delay + self.partition_delay[model_index][partition_index]
        # TODO: get the last finish time of the input partitions if the num. of partitions is > 1
        return partition_finish_time

# TODO: get the first available partition slot with the least amount of waiting time
    def calculate_waiting_delay(self, device_index, slot_index, model_index, partition_index):
        # get finish time of the last partition of the same model
        if partition_index == 0:
            last_finish_time = 0.0
        if partition_index > 0:
            last_finish_time = self.partition_finish_time[model_index][partition_index-1] 
        model_i, partition_i = self.device_partition_schedule[device_index][slot_index] # get the finish time of the slot
        waiting_delay = self.partition_finish_time[model_i][partition_i] - last_finish_time
        if waiting_delay < 0:
            waiting_delay = 0.0
        return waiting_delay

# TODO: finish partition rescheduling based alg. 3
    def reschedule_partition(self, model_index, partition_index):
        # Input: model index (int), layer index (int), number of partitions (int)
        # Output: a device and queue index pair (list) to relocate the partition
        earliest_queue_index = None
        earliest_placement_time = 8.0 # just some random number of seconds that's too long
        for device_index, _ in enumerate(config.local_device_info):
            queue_index_of_last_partition = self.get_prev_slot_device_queue(model_index, partition_index, device_index) 
            if queue_index_of_last_partition == None:
                return [device_index, 0]
            [model_i, partition_i] = self.device_partition_schedule[device_index][queue_index_of_last_partition]
            if self.partition_finish_time[model_i][partition_i] < earliest_placement_time:
                if partition_index > 0 and self.partition_finish_time[model_index][partition_index-1]:
                    earliest_queue_index = queue_index_of_last_partition
        return [device_index, earliest_queue_index]
    
    def reschedule_partitions(self):
        for device_index, device_partition_schedule in enumerate(self.device_partition_schedule):
            for device_queue_index, [model_i, partition_i] in enumerate(self.device_partition_schedule[device_index]):
                [earliest_device_index, earliest_queue_index] = self.reschedule_partition(model_i, partition_i)
                # 1. change the partition schedule
                print('earliest_queue_index: ', earliest_queue_index)
                print('self.device_partition_schedule[earliest_device_index]: ',self.device_partition_schedule[earliest_device_index])
                print('self.device_partition_schedule[earliest_device_index][earliest_queue_index:]: ',self.device_partition_schedule[earliest_device_index][earliest_queue_index:])
                # relocate the partiton
                self.device_partition_schedule[earliest_device_index] = self.device_partition_schedule[earliest_device_index][:earliest_queue_index] + [[model_i, partition_i]] + self.device_partition_schedule[earliest_device_index][earliest_queue_index:]
                input()
                # TODO 2. change the layer schedule
                
                # TODO 3. change partition finish time and delay

##########################################################################################################################################################################################
    
    def create_init_model_partition(self, device_usage_threshold=0.400): # create init partitions based on the time usage limit
        device_usage_threshold_timeslot = int(device_usage_threshold / self.len_timeslot)
        for model_index, models in enumerate(self.service_info):
            print('Model %d'%model_index)
            layer_index = 0
            partition_index = 0
            for device_index, device_schedule in enumerate(self.device_layer_schedule):
                layer_index_in_partition = 0 
                # begin a partition
                # self.layers_in_partitions[model_index].append([])
                # self.partition_delay[model_index].append(0.0)
                if self.service_info[model_index]['layers'][layer_index]['layer_type'] == 'fc':
                    optimal_num_partitions = 1
                else:
                    optimal_num_partitions, input_partition_size = self.get_num_input_partitions(model_index, layer_index)
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
                    # TODO: add the layer to the schedule with adjusted num. of timeslots taken by the layer(?)
                    # timeslot_taken = math.ceil(self.calculate_partition_delay(
                    #                         model_index=model_index,
                    #                         partition_index=partition_index,
                    #                         num_input_partitions=1
                    #                         ) / self.len_timeslot)                           
                    # for t in range(timeslot_taken):
                    # print('self.device_layer_schedule[device_index][eta]: ',self.device_layer_schedule[device_index][eta])
                    self.device_layer_schedule[device_index][eta] = [model_index, layer_index]
                    layer_index += 1
                    print('device_layer_schedule: ',self.device_layer_schedule)
                    print('partitions: ',self.layers_in_partitions)
                    print('partition_delay: ',self.partition_delay)
                    # input()
                self.partition_finish_time[model_index][partition_index] = self.get_partition_finish_time(model_index, partition_index, device_index) # This only works for when initially creating the partition
                # FIXME: device_partition_schedule
                slot_index = self.get_etas_partition_device(model_index, partition_index)
                self.device_partition_schedule[device_index][slot_index] = [model_index, partition_index] # TODO: get the last partiton index within the device that's being used -> get_etas_partition_device()
                print('self.device_partition_schedule: ',self.device_partition_schedule)
                # input()
                print('self.partition_finish_time: ',self.partition_finish_time)
                partition_index += 1
        return self.get_finish_time()
ds = InputDataSet()
# print('Total Timeslots Taken: ', ds.create_model_partitions())
print('Finish time before re-scheduling:',ds.create_init_model_partition())
# print('Finish time after rescheduling: ',ds.reschedule_partitions())