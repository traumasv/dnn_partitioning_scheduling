# Require: a graph of partitioned DNN blocks for each RoI, from each frame (timeslot) from each video stream (device), transmission table
# Ensure: allocation table T_{Y} with the task placement decision and order
import numpy as np
import networkx as nx
import config_smart_camera as config
MOBILENET_INPUT_SIZE = [224*224*3, 112*112*32, 112*112*16, 56*56*24, 28*28*32, 14*14*64, 14*14*96, 7*7*160, 7*7*320, 7*7*1280, 1*1*1280]
OVERHEAD_PER_INPUT = config.
class RequestSet():
    def __init__(self, num_rois:int, num_partitions:int, num_devices:int): 
        # number of RoIs here indicate the number of RoIs across all streams at timestamp t
        # number of partitions indicate the partitions per model
        self.roi_origin = np.randint(low=0, high=num_devices, size=(num_rois)) # elements represent the device index location of input origin
        self.num_devices = num_devices
        self.partition_graph = nx.Graph()
        self.allocation_table = np.full(shape=(num_rois, num_partitions), fill_value=-1) # elements represent the device index where the partition is processed
        self.input_size_table = np.full(shape=(num_rois, num_partitions), fill_value=MOBILENET_INPUT_SIZE) # elements represent the size of the input data for each partition
        self.rank_table = np.full(shape=(num_rois, num_partitions), fill_value=0)
        self.start_time = np.full(shape=self.allocation_table.shape, fill_value=-1)
        self.finish_time = np.full(shape=self.allocation_table.shape, fill_value=-1) 
    
    def get_comp_delay(self) -> float:
        pass

    def get_trans_delay(self) -> float:
        pass

    def get_average_comp_delay(self, roi_index, partition_index) -> float:
        self.input_size_table[roi_index][partition_index] * 

    def rank_upward(self):
        # TODO define upward rank
        for q, roi in enumerate(self.allocation_table):
            for b in range(roi.size, 0, -1):
                partition_rank = self.get_average_comp_delay()
                if b < roi.size - 1:
                    self.rank_table[q][b] = partition_rank + self.rank_table[q][b+1]
                elif b == roi.size - 1:
                    self.rank_table[q][b] = partition_rank
            # TODO get the rank of the past partition
            pass
    def schedule_t(self, partition_table, devices):
        # create a table of partition-decision pair
        for q, partition_set in enumerate(partition_table):
            self.rank_upward(partition_set=partition_set)