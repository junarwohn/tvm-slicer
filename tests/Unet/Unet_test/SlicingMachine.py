import json
import copy
from pyexpat import model
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

    def slice_graph(self, start_nodes, end_nodes, is_quantize_sliced=False):
        graph_config = copy.deepcopy(self.graph_config)

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
            # print(input_lists)
            for input_node_index in input_lists:
                # print(cur_node_index, "->", input_node_index[0])
                mark_list = dfs(input_node_index[0], upper_bound, mark_list)
            return mark_list

        self.sliced_graph = []

        start_points = [start_node - 1 for start_node in start_nodes]
        # start_points.sort(reverse=True)
        end_points = [end_node for end_node in end_nodes]
        # start_points.sort(reverse=True)

        pre_nodes = []
        for start_point in start_points:
            nodes = np.array(sorted(dfs(start_point, 0, [])))
            pre_nodes = np.union1d(pre_nodes, nodes).astype(int)
            # pre_nodes = np.array(sorted(dfs(start_point, 0, [])))
        target_nodes = []
        for end_point in end_points:
            nodes = np.array(sorted(dfs(end_point, 0, [])))
            target_nodes = np.union1d(target_nodes, nodes).astype(int)
            # target_nodes = np.array(sorted(dfs(end_p, 0, [])))
        # print(target_nodes)
        # print("total_nodes")
        total_nodes = [i for i in range(len(graph_config['nodes']))]

        # model_nodes = target_nodes - pre_nodes 
        model_nodes = np.setdiff1d(target_nodes, pre_nodes)
        np.sort(model_nodes)
        
        # complement_nodes = total_nodes - model_nodes
        complement_nodes = pre_nodes
        # complement_nodes = np.setdiff1d(total_nodes, model_nodes)
        # complement_nodes = np.setdiff1d(total_nodes, target_nodes)

        # complement_nodes = total_nodes - model_nodes
        np.sort(complement_nodes)

        # ----------------------------------------
        # Check dependency of input nodes of model
        # ----------------------------------------

        intermediate_nodes = []
        # dep_input_info = defaultdict(list) # dep_node : model_nodes
        input_dependency = defaultdict(list) # input_node : model_node

        #####################
        # print("##############################")
        # print("Initial models")
        # print("pre_nodes", pre_nodes)
        # print("target_nodes", target_nodes)
        # print("model_nodes", model_nodes)
        # print("complement_nodes", complement_nodes)
        # print("##############################")

        for mnode in model_nodes:
            # Get all input nodes in 
            input_nodes = [e[0] for e in graph_config['nodes'][mnode]['inputs']]
            for inode in input_nodes:
                # if there is a input that is not included in model_nodes
                if inode not in model_nodes:
                    # if there is a input that is included in complement_nodes
                    if inode in complement_nodes:
                        input_input_nodes = [e[0] for e in graph_config['nodes'][inode]['inputs']]
                        ######## Quantized node check #########
                        if len(input_input_nodes) != 0 and graph_config["attrs"]["dltype"][1][inode] != 'int8':
                            for iinode in input_input_nodes:
                                iinode_dtype = graph_config["attrs"]["dltype"][1][iinode]
                                iinode_op = graph_config['nodes'][iinode]['op']
                                if iinode_dtype == 'int8' and iinode_op != 'null':
                                    intermediate_nodes.append(inode)
                                    input_dependency[iinode].append(inode)
                                else:
                                    input_dependency[inode].append(mnode)
                        else:
                            input_dependency[inode].append(mnode)

                    # Error : the dependency from nowhere!
                    else:
                        print("Unidentified dependency")
                

        # if len(intermediate_nodes) != 0:
        #     model_nodes = np.concatenate([model_nodes, np.array(intermediate_nodes)])
        #     np.sort(model_nodes)


        # ----------------------------------------
        # Check dependency of output nodes of model
        # ----------------------------------------

        # complement_nodes = post_nodes - model_nodes 
        complement_nodes = np.setdiff1d(total_nodes, target_nodes)
        complement_nodes = np.setdiff1d(complement_nodes, model_nodes)
        complement_nodes = np.setdiff1d(complement_nodes, intermediate_nodes)

        np.sort(complement_nodes)
        # print(complement_nodes)

        # print("######################################")
        # print("After input analyze")
        # print("intermediate_nodes", intermediate_nodes)
        # print("input_dependency", input_dependency)
        # print("model_nodes", model_nodes)
        # print("######################################")

        # Check dependency nodes of ex nodes - for output nodes in model nodes.
        # dep_output_info = defaultdict(list) # model_nodes : ex_dep_node
        output_dependency = defaultdict(list) # model_node : input_node
        for cnode in complement_nodes:
            input_nodes = [e[0] for e in graph_config['nodes'][cnode]['inputs']]
            for inode in input_nodes:
                if inode in model_nodes:
                    ######## Quantized node check #########
                    # Check the input of input. 
                    # If input of input is int8 -> We assume that that node has been quantized 
                    # We should export that node instead of orignally dependent node.
                    input_input_nodes = [e[0] for e in graph_config['nodes'][inode]['inputs']]
                    if len(input_input_nodes) != 0:
                        for iinode in input_input_nodes:
                            if iinode in model_nodes:
                                iinode_dtype = graph_config["attrs"]["dltype"][1][iinode]
                                iinode_op = graph_config['nodes'][iinode]['op']
                                if iinode_dtype == 'int8' and iinode_op != 'null' and graph_config["attrs"]["dltype"][1][inode] != 'int8':
                                    # print("######################")
                                    # print(input_input_nodes)
                                    # print("######################")
                                    # intermediate_nodes.append(inode)
                                    output_dependency[iinode].append(cnode)
                                else:
                                    output_dependency[inode].append(cnode)
                    else:
                        output_dependency[inode].append(cnode)

        # print("######################################")
        # print("After output analyze")
        # print("output_dependency", output_dependency)
        # print("model_nodes", model_nodes)
        # print("######################################")

        if len(intermediate_nodes) != 0:
            model_nodes = np.concatenate([model_nodes, np.array(intermediate_nodes)])
            np.sort(model_nodes)

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


        # print("input_dependency.keys()", input_dependency.keys())
        # Add input
        input_nodes = sorted(input_dependency.keys())

        for input_node_index in input_nodes:
            input_node_info = copy.deepcopy(graph_config['nodes'][input_node_index])
            input_node_info["op"] = "null"
            input_node_info["name"] = "input_{}".format(input_node_index)
            input_node_info["inputs"] = [] 
            sliced_graph_config["nodes"].append(input_node_info)
            sliced_graph_config["arg_nodes"].append(int(input_nodes.index(input_node_index)))
            sliced_graph_config["attrs"]["dltype"][1].append(graph_config["attrs"]["dltype"][1][input_node_index])
            sliced_graph_config["attrs"]["device_index"][1].append(graph_config["attrs"]["device_index"][1][input_node_index])
            sliced_graph_config["attrs"]["storage_id"][1].append(graph_config["attrs"]["storage_id"][1][input_node_index])
            sliced_graph_config["attrs"]["shape"][1].append(copy.deepcopy(graph_config["attrs"]["shape"][1][input_node_index]))
            sliced_graph_config["node_row_ptr"].append(int(input_node_index))


        # Add body
        model_nodes = sorted(model_nodes)
        model_nodes = input_nodes + model_nodes
        # print("model_nodes", model_nodes)

        for node_index in model_nodes[len(input_nodes):]:
            sliced_graph_config["nodes"].append(copy.deepcopy(graph_config['nodes'][node_index]))
            if graph_config["nodes"][node_index]["op"] == "null":
                sliced_graph_config["arg_nodes"].append(int(model_nodes.index(node_index)))
            sliced_graph_config["attrs"]["dltype"][1].append(graph_config["attrs"]["dltype"][1][node_index])
            sliced_graph_config["attrs"]["device_index"][1].append(graph_config["attrs"]["device_index"][1][node_index])
            sliced_graph_config["attrs"]["storage_id"][1].append(graph_config["attrs"]["storage_id"][1][node_index])
            sliced_graph_config["attrs"]["shape"][1].append(copy.deepcopy(graph_config["attrs"]["shape"][1][node_index]))
            sliced_graph_config["node_row_ptr"].append(int(node_index))

        # Set output
        output_nodes = sorted(output_dependency.keys())
        # When this chunck contain the tail of original model
        # TODO : Add logic for originally multiple output model.
        if len(output_nodes) == 0:
            output_nodes = [len(graph_config['nodes']) - 1]
        # print("output_dependency", output_dependency)
        # print("output_nodes", output_nodes)
        # Lookup Table for indexing.
        # {original_index : node_name}
        original_lut = dict()
        for idx, node_info in enumerate(graph_config['nodes']):
            name = node_info['name']
            original_lut[idx] = name

        # Lookup Table for indexing.
        # {node_name : new_index}
        lut = dict()
        for idx, node_info in enumerate(sliced_graph_config['nodes']):
            name = node_info['name']
            lut[name] = idx

        # print(lut)
        # Set input
        for node_index, input_index in enumerate(model_nodes):
            node_input_indexs = sliced_graph_config["nodes"][node_index]['inputs']
            for i, node in enumerate(node_input_indexs):
                try:
                    sliced_graph_config["nodes"][node_index]['inputs'][i] = [lut[original_lut[node[0]]], 0, 0]
                # when required node is transformed into input_{} 
                except:
                    # print('input_{}'.format(node[0]))
                    # print('name' , sliced_graph_config["nodes"][node_index]['name'])
                    sliced_graph_config["nodes"][node_index]['inputs'][i] = [lut['input_{}'.format(node[0])], 0, 0]
                    # except:
                    #     # parent int8
                    #     if 'input_{}'.format(node[0] - 1) in lut:
                    #         sliced_graph_config["nodes"][node_index]['inputs'][i] = [lut['input_{}'.format(node[0] - 1)], 0, 0]
                    #     else:
                    #         print("error")
                    #         print(lut)
                    #         continue
        output_nodes = np.setdiff1d(output_nodes, np.array([0]))
        for output_node_index in output_nodes:
            try:
                sliced_graph_config["heads"].append([lut[original_lut[output_node_index]], 0, 0])
            except:
                # print('input_{}'.format(node[0]))
                sliced_graph_config["heads"].append([lut['input_{}'.format(output_node_index)], 0, 0])


            # sliced_graph_config["heads"].append([output_node_index + len(input_nodes), 0, 0])

        # if len(sliced_graph_config["heads"]) == 0:
        #     sliced_graph_config["heads"].append([len(model_nodes) - 1, 0, 0])
        # Set rest : node_row_ptr
        sliced_graph_config["node_row_ptr"] = [i for i in range(len(sliced_graph_config["nodes"]) + 1)]
        # print("====================")

        # return [sliced_graph_config, input_nodes, [o + len(input_nodes) for o in output_nodes]]
        # print(output_dependency.keys())
        # print([h[0] for h in sliced_graph_config["heads"]])
        # print(output_nodes)
        return [sliced_graph_config, input_nodes, output_nodes.tolist()]


    def get_all_intermediate_node(self):
        intermediate_nodes = []
        for idx, node in enumerate(self.graph_config['nodes']):
            if len(node['inputs']) != 0:
                intermediate_nodes.append(idx)
        return intermediate_nodes


