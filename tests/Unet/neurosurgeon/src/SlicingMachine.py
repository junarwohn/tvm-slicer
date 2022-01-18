import json
import copy
import numpy as np
from re import M
from sys import excepthook
from collections import defaultdict
from threading import currentThread

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


# RULE : return outputs by node number order
# return output node info and input node info (original node number)
# [ [graph1, [input node ("{original node},,,")], [output node] "{original node,,,}"], [graph2, [input node], [output node]],,, ]
# Sliced by range (start, end]
class TVMSlicer:
    #def __init__(self, graph_config='', slicing_point=[[]]):
    def __init__(self, graph_config=''):
        if isinstance(graph_config, str):
            try:
                graph_config = json.loads(graph_config)  
            except:
                return
        self.graph_config = copy.deepcopy(graph_config)
        # if len(slicing_point) < 2:
            # raise Exception("[SlicingMachine] slicing_point should have at least 3 points (model start, model end")

        # for point in slicing_point[1:-1]:
            # if len(graph_config['nodes'][point]['inputs']) == 0:
                # raise Exception("[SlicingMachine] Node {} is not Intermediate Node. slicing point should be intermediate point".format(point))

        #def dfs(cur_node_index, upper_bound, mark_list):
        #    # Already visited
        #    if cur_node_index in mark_list:
        #        return mark_list

        #    # Check upper bound
        #    if cur_node_index == upper_bound:
        #        mark_list.append(cur_node_index)
        #        return mark_list

        #    # Traverse
        #    mark_list.append(cur_node_index)
        #    input_lists = graph_config['nodes'][cur_node_index]['inputs']
        #    for input_node_index in input_lists:
        #        mark_list = dfs(input_node_index[0], upper_bound, mark_list)
        #    return mark_list
        #self.sliced_graph = []

        ## self.dfs_list = dfs(11, 0, [])
        ## print(dfs(7, 0, []))
        ## print(dfs(11, 0, []))
        ## print(np.setdiff1d(dfs(11, 0, []),dfs(0, 0, [])))
        #for start_p, end_p in slicing_point:
        #    pre_nodes = np.array(sorted(dfs(start_p, 0, [])))
        #    target_nodes = np.array(sorted(dfs(end_p, 0, [])))
        #    total_nodes = [i for i in range(len(graph_config['nodes']))]

        #    # model_nodes = target_nodes - pre_nodes 
        #    model_nodes = np.setdiff1d(target_nodes, pre_nodes)
        #    np.sort(model_nodes)

        #    # Check dependency nodes of model nodes - for input nodes in model nodes
        #    dep_input_info = defaultdict(list) # dep_node : model_nodes
        #    for t_node in model_nodes:
        #        for input_node_index in graph_config['nodes'][t_node]['inputs']:
        #            if input_node_index[0] not in model_nodes:
        #                dep_input_info[input_node_index[0]].append(t_node)
        #    
        #    # ex_nodes = post_nodes - model_nodes 
        #    ex_nodes = np.setdiff1d(total_nodes, model_nodes)
        #    np.sort(ex_nodes)

        #    # Check dependency nodes of ex nodes - for output nodes in model nodes.
        #    dep_output_info = defaultdict(list) # model_nodes : ex_dep_node
        #    for e_node in ex_nodes:
        #        for input_node_index in graph_config['nodes'][e_node]['inputs']:
        #            if input_node_index[0] in model_nodes:
        #                dep_output_info[input_node_index[0]].append(e_node)
        #    

        #    sliced_graph_config = {
        #        "nodes" : [],
        #        "arg_nodes": [],
        #        "heads": [],
        #        "attrs": { 
        #            "dltype": [
        #                "list_str",
        #                []
        #            ],
        #            "device_index": [
        #                "list_int",
        #                []
        #            ],
        #            "storage_id": [
        #                "list_int",
        #                []
        #            ],
        #            "shape": [
        #                "list_shape",
        #                []
        #            ],
        #        },
        #    "node_row_ptr": []
        #    }


        #    # Add input
        #    input_nodes = sorted(dep_input_info.keys())
        #    # print(input_nodes)
        #    for input_node_index in input_nodes:
        #        sliced_graph_config["nodes"].append(copy.deepcopy(graph_config['nodes'][input_node_index]))
        #        sliced_graph_config["nodes"][-1]["op"] = "null"
        #        sliced_graph_config["nodes"][-1]["name"] = "input_{}".format(input_node_index)
        #        sliced_graph_config["nodes"][-1]["inputs"] = [] 
        #        sliced_graph_config["arg_nodes"].append(int(input_nodes.index(input_node_index)))
        #        sliced_graph_config["attrs"]["dltype"][1].append(graph_config["attrs"]["dltype"][1][input_node_index])
        #        sliced_graph_config["attrs"]["device_index"][1].append(graph_config["attrs"]["device_index"][1][input_node_index])
        #        sliced_graph_config["attrs"]["storage_id"][1].append(graph_config["attrs"]["storage_id"][1][input_node_index])
        #        sliced_graph_config["attrs"]["shape"][1].append(copy.deepcopy(graph_config["attrs"]["shape"][1][input_node_index]))
        #        sliced_graph_config["node_row_ptr"].append(int(input_node_index))

        #    # Add body
        #    model_nodes = sorted(model_nodes)
        #    model_nodes = input_nodes + model_nodes
        #    # print(model_nodes)
        #    for node_index in model_nodes[len(input_nodes):]:
        #        sliced_graph_config["nodes"].append(copy.deepcopy(graph_config['nodes'][node_index]))
        #        if graph_config["nodes"][node_index]["op"] == "null":
        #            sliced_graph_config["arg_nodes"].append(int(model_nodes.index(node_index)))
        #        sliced_graph_config["attrs"]["dltype"][1].append(graph_config["attrs"]["dltype"][1][node_index])
        #        sliced_graph_config["attrs"]["device_index"][1].append(graph_config["attrs"]["device_index"][1][node_index])
        #        sliced_graph_config["attrs"]["storage_id"][1].append(graph_config["attrs"]["storage_id"][1][node_index])
        #        sliced_graph_config["attrs"]["shape"][1].append(copy.deepcopy(graph_config["attrs"]["shape"][1][node_index]))
        #        sliced_graph_config["node_row_ptr"].append(int(node_index))

        #    # print(graph_config)
        #    # Set input
        #    for node_index, input_index in enumerate(model_nodes):
        #        node_input_indexs = sliced_graph_config["nodes"][node_index]['inputs']
        #        for i, node in enumerate(node_input_indexs):
        #            sliced_graph_config["nodes"][node_index]['inputs'][i] = [model_nodes.index(node[0]), 0, 0]

        #    # for dep_input_index in dep_input_info:
        #    #     dep_node_indexs = dep_input_info[dep_input_index]
        #    #     for node_index in dep_node_indexs:
        #    #         # print(sliced_graph_config["nodes"][model_nodes.index(node_index)]['inputs'])
        #    #         node_input_index = sliced_graph_config["nodes"][model_nodes.index(node_index)]['inputs'].index([dep_input_index, 0, 0])
        #    #         sliced_graph_config["nodes"][model_nodes.index(node_index)]['inputs'][node_input_index] = [input_nodes.index(dep_input_index), 0, 0]

        #    # Set output
        #    output_nodes = sorted(dep_output_info.keys())
        #    if len(output_nodes) == 0:
        #        output_nodes = [len(model_nodes) - 1]

        #    # print(output_nodes)
        #    for output_node_index in output_nodes:
        #        sliced_graph_config["heads"].append([output_node_index, 0, 0])

        #    # if len(sliced_graph_config["heads"]) == 0:
        #    #     sliced_graph_config["heads"].append([len(model_nodes) - 1, 0, 0])
        #    # Set rest : node_row_ptr
        #    sliced_graph_config["node_row_ptr"] = [i for i in range(len(sliced_graph_config["nodes"]) + 1)]

        #    self.sliced_graph.append([sliced_graph_config, input_nodes, output_nodes])
        ## LAST sort heads

    def get_inputs(self):
        return [[i, g] for i, g in enumerate(zip(self.group, self.front_req, self.back_req))]

    def get_graph(self):
        return self.sliced_graph

    def get_mark(self):
        return self.dfs_list

    def slice_graph(self, start_node, end_node):

        graph_config = copy.deepcopy(self.graph_config)
        #print(len(graph_config['nodes']))
        def dfs(cur_node_index, upper_bound, mark_list):
            # Already visited
            if cur_node_index in mark_list:
                return mark_list

            # Check upper bound
            if cur_node_index == upper_bound:
                mark_list.append(cur_node_index)
                return mark_list

            # Traverse
            mark_list.append(cur_node_index)
            input_lists = graph_config['nodes'][cur_node_index]['inputs']
            for input_node_index in input_lists:
                mark_list = dfs(input_node_index[0], upper_bound, mark_list)
            return mark_list
        self.sliced_graph = []

        # self.dfs_list = dfs(11, 0, [])
        # print(dfs(7, 0, []))
        # print(dfs(11, 0, []))
        # print(np.setdiff1d(dfs(11, 0, []),dfs(0, 0, [])))
        #for start_p, end_p in slicing_point:
        start_p = start_node
        end_p = end_node

        pre_nodes = np.array(sorted(dfs(start_p, 0, [])))
        target_nodes = np.array(sorted(dfs(end_p, 0, [])))
        total_nodes = [i for i in range(len(graph_config['nodes']))]

        # model_nodes = target_nodes - pre_nodes 
        model_nodes = np.setdiff1d(target_nodes, pre_nodes)
        np.sort(model_nodes)

        # Check dependency nodes of model nodes - for input nodes in model nodes
        dep_input_info = defaultdict(list) # dep_node : model_nodes
        for t_node in model_nodes:
            for input_node_index in graph_config['nodes'][t_node]['inputs']:
                if input_node_index[0] not in model_nodes:
                    dep_input_info[input_node_index[0]].append(t_node)
        
        # ex_nodes = post_nodes - model_nodes 
        ex_nodes = np.setdiff1d(total_nodes, model_nodes)
        np.sort(ex_nodes)

        # Check dependency nodes of ex nodes - for output nodes in model nodes.
        dep_output_info = defaultdict(list) # model_nodes : ex_dep_node
        for e_node in ex_nodes:
            for input_node_index in graph_config['nodes'][e_node]['inputs']:
                if input_node_index[0] in model_nodes:
                    dep_output_info[input_node_index[0]].append(e_node)
        

        sliced_graph_config = {
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


        # Add input
        input_nodes = sorted(dep_input_info.keys())
        # print(input_nodes)
        for input_node_index in input_nodes:
            sliced_graph_config["nodes"].append(copy.deepcopy(graph_config['nodes'][input_node_index]))
            sliced_graph_config["nodes"][-1]["op"] = "null"
            sliced_graph_config["nodes"][-1]["name"] = "input_{}".format(input_node_index)
            sliced_graph_config["nodes"][-1]["inputs"] = [] 
            sliced_graph_config["arg_nodes"].append(int(input_nodes.index(input_node_index)))
            sliced_graph_config["attrs"]["dltype"][1].append(graph_config["attrs"]["dltype"][1][input_node_index])
            sliced_graph_config["attrs"]["device_index"][1].append(graph_config["attrs"]["device_index"][1][input_node_index])
            sliced_graph_config["attrs"]["storage_id"][1].append(graph_config["attrs"]["storage_id"][1][input_node_index])
            sliced_graph_config["attrs"]["shape"][1].append(copy.deepcopy(graph_config["attrs"]["shape"][1][input_node_index]))
            sliced_graph_config["node_row_ptr"].append(int(input_node_index))

        # Add body
        model_nodes = sorted(model_nodes)
        model_nodes = input_nodes + model_nodes
        # print(model_nodes)
        for node_index in model_nodes[len(input_nodes):]:
            sliced_graph_config["nodes"].append(copy.deepcopy(graph_config['nodes'][node_index]))
            if graph_config["nodes"][node_index]["op"] == "null":
                sliced_graph_config["arg_nodes"].append(int(model_nodes.index(node_index)))
            sliced_graph_config["attrs"]["dltype"][1].append(graph_config["attrs"]["dltype"][1][node_index])
            sliced_graph_config["attrs"]["device_index"][1].append(graph_config["attrs"]["device_index"][1][node_index])
            sliced_graph_config["attrs"]["storage_id"][1].append(graph_config["attrs"]["storage_id"][1][node_index])
            sliced_graph_config["attrs"]["shape"][1].append(copy.deepcopy(graph_config["attrs"]["shape"][1][node_index]))
            sliced_graph_config["node_row_ptr"].append(int(node_index))

        # print(graph_config)
        # Set input
        for node_index, input_index in enumerate(model_nodes):
            node_input_indexs = sliced_graph_config["nodes"][node_index]['inputs']
            for i, node in enumerate(node_input_indexs):
                sliced_graph_config["nodes"][node_index]['inputs'][i] = [model_nodes.index(node[0]), 0, 0]

        # for dep_input_index in dep_input_info:
        #     dep_node_indexs = dep_input_info[dep_input_index]
        #     for node_index in dep_node_indexs:
        #         # print(sliced_graph_config["nodes"][model_nodes.index(node_index)]['inputs'])
        #         node_input_index = sliced_graph_config["nodes"][model_nodes.index(node_index)]['inputs'].index([dep_input_index, 0, 0])
        #         sliced_graph_config["nodes"][model_nodes.index(node_index)]['inputs'][node_input_index] = [input_nodes.index(dep_input_index), 0, 0]

        # Set output
        output_nodes = sorted(dep_output_info.keys())
        if len(output_nodes) == 0:
            output_nodes = [len(model_nodes) - 1]

        # print(output_nodes)
        for output_node_index in output_nodes:
            sliced_graph_config["heads"].append([output_node_index, 0, 0])

        # if len(sliced_graph_config["heads"]) == 0:
        #     sliced_graph_config["heads"].append([len(model_nodes) - 1, 0, 0])
        # Set rest : node_row_ptr
        sliced_graph_config["node_row_ptr"] = [i for i in range(len(sliced_graph_config["nodes"]) + 1)]

        return [sliced_graph_config, input_nodes, output_nodes]


    def get_all_intermediate_node(self):
        intermediate_nodes = []
        for idx, node in enumerate(self.graph_config['nodes']):
            if len(node['inputs']) != 0:
                intermediate_nodes.append(idx)
        return intermediate_nodes


