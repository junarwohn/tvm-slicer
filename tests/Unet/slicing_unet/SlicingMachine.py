import json
import copy
from re import M
from sys import excepthook
from collections import defaultdict

# Graph Json Structure
#
# nodes
# arg_nodes
# heads
# attrs
# - dltype
# - device_index
# - storage_id
# - shape
# node_row_ptr


class TVMSlicer:
    def __init__(self, graph_config='', slicing_point=0):
        if isinstance(graph_config, str):
            try:
                graph_config = json.loads(graph_config)  
            except:
                return
        if slicing_point == 0:
            return json.dumps(graph_config)

        group = ['front' for i in range(slicing_point)] + ['back' for i in range(len(graph_config['nodes']) - slicing_point)]

        # repeat check until no non-front_req && back_req
        while(True):
            move_cnt = 0
            front_req = [False for i in range(len(group))]
            back_req = [False for i in range(len(group))]
            for node_idx, node_info in enumerate(graph_config['nodes']):
                input_nodes = [n[0] for n in node_info['inputs']]
                if group[node_idx] == 'front':
                    for input_node in input_nodes:
                        front_req[input_node] = True
                else:
                    for input_node in input_nodes:
                        back_req[input_node] = True
            for node_idx, (f, b) in enumerate(zip(front_req, back_req)):
                if not f and b and graph_config['nodes'][node_idx]['op'] == 'null' and group[node_idx] == 'front': 
                    group[node_idx] = 'back'
                    move_cnt += 1
            if move_cnt == 0:
                break

        front_node_idxs = []
        front_output_idxs = []
        back_node_idxs = []
        back_input_node_idxs = []
        for node_idx, (f, b) in enumerate(zip(front_req, back_req)):
            if group[node_idx] == 'front':
                front_node_idxs.append(node_idx)
                if b:
                    front_output_idxs.append(node_idx)
                    back_input_node_idxs.append(node_idx)
            else:
                back_node_idxs.append(node_idx)

        # Front
        front_graph_config = {
            "nodes" : [],
            "arg_nodes": [],
            "heads": [],
            "attrs": { 
                "dltype": [
                    "list_str",
                    []
                ],
                "device_index": [
                    "list_int",
                    []
                ],
                "storage_id": [
                    "list_int",
                    []
                ],
                "shape": [
                    "list_shape",
                    []
                ],
            },
           "node_row_ptr": []
        }

        for fidx in front_node_idxs:
            node = copy.deepcopy(graph_config['nodes'][fidx])
            dltype = graph_config['attrs']['dltype'][1][fidx]
            device_index = graph_config['attrs']['device_index'][1][fidx]
            storage_id = graph_config['attrs']['storage_id'][1][fidx]
            shape = graph_config['attrs']['shape'][1][fidx]
            if node['op'] != 'null':
                inputs = [i[0] for i in node['inputs']]
                inputs = [[front_node_idxs.index(k), 0, 0] for k in inputs]
                node['inputs'] = inputs
            front_graph_config['nodes'].append(node)
            front_graph_config['attrs']['dltype'][1].append(dltype)
            front_graph_config['attrs']['device_index'][1].append(device_index)
            front_graph_config['attrs']['storage_id'][1].append(storage_id)
            front_graph_config['attrs']['shape'][1].append(shape)

        for curidx, fidx in enumerate(front_node_idxs):
            if front_graph_config['nodes'][curidx]['op'] == 'null':
                front_graph_config['arg_nodes'].append(curidx)
            front_graph_config['node_row_ptr'].append(curidx)

        front_graph_config['node_row_ptr'].append(len(front_graph_config['node_row_ptr']))

        for oidx in front_output_idxs:
            front_graph_config['heads'].append([front_node_idxs.index(oidx), 0, 0])

        self.front_graph_config = front_graph_config

        # Back
        back_graph_config = {
            "nodes" : [],
            "arg_nodes": [],
            "heads": [],
            "attrs": { 
                "dltype": [
                    "list_str",
                    []
                ],
                "device_index": [
                    "list_int",
                    []
                ],
                "storage_id": [
                    "list_int",
                    []
                ],
                "shape": [
                    "list_shape",
                    []
                ],
            },
           "node_row_ptr": []
        }

        for curidx, bidx in enumerate(back_input_node_idxs):
            node = copy.deepcopy(graph_config['nodes'][bidx]) 
            dltype = graph_config['attrs']['dltype'][1][bidx]
            device_index = graph_config['attrs']['device_index'][1][bidx]
            storage_id = graph_config['attrs']['storage_id'][1][bidx]
            shape = graph_config['attrs']['shape'][1][bidx]
            node['op'] = 'null'
            node['name'] = 'input_{}'.format(curidx + 1)
            node['inputs'] = []
            back_graph_config['nodes'].append(node)
            back_graph_config['attrs']['dltype'][1].append(dltype)
            back_graph_config['attrs']['device_index'][1].append(device_index)
            back_graph_config['attrs']['storage_id'][1].append(storage_id)
            back_graph_config['attrs']['shape'][1].append(shape)

        back_node_idxs = back_input_node_idxs + back_node_idxs

        for curidx, bidx in enumerate(back_node_idxs[len(back_input_node_idxs):]):
            node = copy.deepcopy(graph_config['nodes'][bidx]) 
            dltype = graph_config['attrs']['dltype'][1][bidx]
            device_index = graph_config['attrs']['device_index'][1][bidx]
            storage_id = graph_config['attrs']['storage_id'][1][bidx]
            shape = graph_config['attrs']['shape'][1][bidx]
            if node['op'] != 'null':
                inputs = [i[0] for i in node['inputs']]
                inputs = [[back_node_idxs.index(k), 0, 0] for k in inputs]
                node['inputs'] = inputs
            back_graph_config['nodes'].append(node)
            back_graph_config['attrs']['dltype'][1].append(dltype)
            back_graph_config['attrs']['device_index'][1].append(device_index)
            back_graph_config['attrs']['storage_id'][1].append(storage_id)
            back_graph_config['attrs']['shape'][1].append(shape)

        for curidx, fidx in enumerate(back_node_idxs):
            if back_graph_config['nodes'][curidx]['op'] == 'null':
                back_graph_config['arg_nodes'].append(curidx)
            back_graph_config['node_row_ptr'].append(curidx)

        back_graph_config['node_row_ptr'].append(len(back_graph_config['node_row_ptr']))

        back_graph_config['heads'].append([len(back_node_idxs) - 1, 0, 0])

        self.back_graph_config = back_graph_config

        # self.group = group
        # self.front_req = front_req
        # self.back_req = back_req

    def get_inputs(self):
        return [[i, g] for i, g in enumerate(zip(self.group, self.front_req, self.back_req))]

    def get_graph(self):
        return self.front_graph_config, self.back_graph_config