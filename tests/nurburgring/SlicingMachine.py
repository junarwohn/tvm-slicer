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

    def get_inputs(self):
        return [[i, g] for i, g in enumerate(zip(self.group, self.front_req, self.back_req))]

    def get_graph(self):
        return self.sliced_graph

    def get_mark(self):
        return self.dfs_list

    def slice_graph(self, start_node, end_node, is_quantize_sliced=False):

        graph_config = copy.deepcopy(self.graph_config)

        def dfs(cur_node_index, upper_bound, mark_list):
            # Already visited
            if cur_node_index in mark_list:
                return mark_list

            # Check upper bound
            if cur_node_index <= upper_bound:
                mark_list.append(cur_node_index)
                return mark_list

            # Traverse
            mark_list.append(cur_node_index)
            input_lists = graph_config['nodes'][cur_node_index]['inputs']
            for input_node_index in input_lists:
                mark_list = dfs(input_node_index[0], upper_bound, mark_list)
            return mark_list

        self.sliced_graph = []

        start_p = start_node - 1
        end_p = end_node

        pre_nodes = np.array(sorted(dfs(start_p, 0, [])))
        target_nodes = np.array(sorted(dfs(end_p, 0, [])))
        total_nodes = [i for i in range(len(graph_config['nodes']))]

        print("====================")
        print("pre_nodes :", pre_nodes)
        print("target_nodes :", target_nodes)
        print("total_nodes :", total_nodes)

        # model_nodes = target_nodes - pre_nodes 
        model_nodes = np.setdiff1d(target_nodes, pre_nodes)
        np.sort(model_nodes)

        
        # ----------------------------------------
        # Check dependency input nodes of model
        # ----------------------------------------

        intermediate_nodes = []
        dep_input_info = defaultdict(list) # dep_node : model_nodes
        for t_node in model_nodes:
            for input_node in graph_config['nodes'][t_node]['inputs']:
                input_node_index = input_node[0]
                if input_node_index not in model_nodes:
                    print('-----------')
                    print(input_node_index)
                    if is_quantize_sliced:
                        parent_nodes = graph_config['nodes'][input_node_index]['inputs']
                        if len(parent_nodes) == 0:
                            dep_input_info[input_node_index].append(t_node)
                        else:
                            for parent_node in parent_nodes:
                                parent_node_index = parent_node[0]
                                parent_node_dtype = graph_config["attrs"]["dltype"][1][parent_node_index]
                                parent_node_op = graph_config['nodes'][input_node_index]['op']
                                if parent_node_dtype == 'int8' and parent_node_op != 'null':
                                    intermediate_nodes.append(input_node_index)
                                    dep_input_info[parent_node_index].append(input_node_index)
                                else:
                                    dep_input_info[input_node_index].append(t_node)
                    else:
                        dep_input_info[input_node_index].append(t_node)
                    print(input_node_index)
                    print(dep_input_info)
                    print('-----------')
                
        if len(intermediate_nodes) != 0:
            model_nodes = np.concatenate([model_nodes, np.array(intermediate_nodes)])
            np.sort(model_nodes)

        # ----------------------------------------
        # Check dependency output nodes of model
        # ----------------------------------------

        # ex_nodes = post_nodes - model_nodes 
        ex_nodes = np.setdiff1d(total_nodes, model_nodes)
        np.sort(ex_nodes)
        print("ex_nodes", ex_nodes)
        # Check dependency nodes of ex nodes - for output nodes in model nodes.
        dep_output_info = defaultdict(list) # model_nodes : ex_dep_node
        for e_node in ex_nodes:
            for input_node_index in graph_config['nodes'][e_node]['inputs']:
                if input_node_index[0] in model_nodes:
                    if is_quantize_sliced:
                        cur_output_node = input_node_index[0]
                        parent_output_node_index = graph_config['nodes'][cur_output_node]['inputs'][0][0]
                        parent_output_node_dtype = graph_config["attrs"]["dltype"][1][parent_output_node_index]
                        if parent_output_node_dtype == 'int8':
                            try:
                                dep_output_info[int(np.where(model_nodes == parent_output_node_index)[0])] = parent_output_node_index[0]
                            except:
                                pass
                        else:
                            dep_output_info[int(np.where(model_nodes == input_node_index[0])[0])] = input_node_index[0]
                    else:
                        dep_output_info[int(np.where(model_nodes == input_node_index[0])[0])] = input_node_index[0]
        print("dep_output_info", dep_output_info)


        # # If the model is quantized to reduce data transmission overhead at sliced node,
        # # parent of the sliced node should be added to model_nodes  

        # dequant_nodes = []
        # if is_quantize_sliced:
        #     # Traverse
        #     for t_node in model_nodes:
        #         for input_node_index in graph_config['nodes'][t_node]['inputs']:
        #             if input_node_index[0] not in model_nodes:
        #                 print("input_node_index[0]", input_node_index[0])
        #                 try:
        #                     # Check if the parent of node has int8 type input
        #                     parent_node_index = input_node_index[0]
        #                     # parent_node_dtype = graph_config["attrs"]["dltype"][1][parent_node_index]
        #                     # print("parent_node_index", parent_node_index)
        #                     parent_node_input_index = graph_config['nodes'][parent_node_index]['inputs'][0][0]
        #                     # # print("parent_node_input_index", parent_node_input_index)
        #                     parent_node_input_dtype = graph_config["attrs"]["dltype"][1][parent_node_input_index]
        #                     # print(parent_node_index, parent_node_input_index)
        #                     if parent_node_input_dtype == 'int8':
        #                         dequant_nodes.append(parent_node_index)
        #                         # # dequant args
        #                         # dequant_input_arg_nodes = graph_config['nodes'][parent_node_input_index]['inputs']
        #                         # for dequant_input_arg_node in dequant_input_arg_nodes:
        #                         #     arg_index = dequant_input_arg_node[0]
        #                         #     if graph_config['nodes'][arg_index]['op'] == 'null':
        #                         #         dequant_nodes.append(arg_index)
        #                         # print("int8 added", input_node_index[0])
        #                     else:
        #                         raise Exception("No int8")                             
        #                 except:
        #                     pass

        #     if len(dequant_nodes) != 0:
        #         # print(dequant_nodes)
        #         model_nodes = np.concatenate([model_nodes, np.array(dequant_nodes)])
        #         np.sort(model_nodes)

        # dep_input_info = defaultdict(list) # dep_node : model_nodes
        # for t_node in model_nodes:
        #     for input_node_index in graph_config['nodes'][t_node]['inputs']:
        #         if input_node_index[0] not in model_nodes:
        #             dep_input_info[input_node_index[0]].append(t_node)
                

        # # ex_nodes = post_nodes - model_nodes 
        # ex_nodes = np.setdiff1d(total_nodes, model_nodes)
        # np.sort(ex_nodes)
        # # Check dependency nodes of ex nodes - for output nodes in model nodes.
        # dep_output_info = defaultdict(list) # model_nodes : ex_dep_node
        # for e_node in ex_nodes:
        #     for input_node_index in graph_config['nodes'][e_node]['inputs']:
        #         if input_node_index[0] in model_nodes and len(dequant_nodes) == 0:
        #             if is_quantize_sliced:
        #                 cur_output_node = input_node_index[0]
        #                 parent_output_node_index = graph_config['nodes'][cur_output_node]['inputs'][0][0]
        #                 parent_output_node_dtype = graph_config["attrs"]["dltype"][1][parent_output_node_index]
        #                 if parent_output_node_dtype == 'int8':
        #                     print("haha")
        #                     # dep_output_info[parent_output_node_index].append(e_node)
        #                     print(model_nodes, parent_output_node_index, np.where(model_nodes == parent_output_node_index))
        #                     dep_output_info[int(np.where(model_nodes == parent_output_node_index)[0])].append(e_node)
        #                 else:
        #                     # dep_output_info[input_node_index[0]].append(e_node)
        #                     dep_output_info[int(np.where(model_nodes == input_node_index[0])[0])].append(e_node)
        #             else:
        #                 # dep_output_info[input_node_index[0]].append(e_node)
        #                 dep_output_info[int(np.where(model_nodes == input_node_index[0])[0])].append(e_node)
        #             # dep_output_info[int(np.where(model_nodes == input_node_index[0])[0])].append(e_node)
        # print("dep_output_info", dep_output_info)
        # # # Check dependency nodes of ex nodes - for output nodes in model nodes.
        # # dep_output_info = defaultdict(list) # model_nodes : ex_dep_node
        # # for e_node in ex_nodes:
        # #     for input_node_index in graph_config['nodes'][e_node]['inputs']:
        # #         if input_node_index[0] in model_nodes:
        # #             dep_output_info[input_node_index[0]].append(e_node)

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

        print("dep_input_info.keys()", dep_input_info.keys())
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
        print("model_nodes", model_nodes)
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
            output_nodes = [len(model_nodes) - 1 - len(input_nodes)]

        # print(output_nodes)
        for output_node_index in output_nodes:
            sliced_graph_config["heads"].append([output_node_index + len(input_nodes), 0, 0])

        # if len(sliced_graph_config["heads"]) == 0:
        #     sliced_graph_config["heads"].append([len(model_nodes) - 1, 0, 0])
        # Set rest : node_row_ptr
        sliced_graph_config["node_row_ptr"] = [i for i in range(len(sliced_graph_config["nodes"]) + 1)]
        print("====================")

        # return [sliced_graph_config, input_nodes, [o + len(input_nodes) for o in output_nodes]]
        return [sliced_graph_config, input_nodes, [int(dep_output_info[o]) for o in sorted(dep_output_info.keys())]]


    def get_all_intermediate_node(self):
        intermediate_nodes = []
        for idx, node in enumerate(self.graph_config['nodes']):
            if len(node['inputs']) != 0:
                intermediate_nodes.append(idx)
        return intermediate_nodes


