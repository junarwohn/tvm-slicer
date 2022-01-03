import json
import copy
from sys import excepthook


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

        front_graph_json_data = copy.deepcopy(graph_config)
        back_graph_json_data = copy.deepcopy(graph_config)

        num_node_dep = 0
        node_arg_dep = []
        node_input_dep = []

        # Traverse and mark

        for node_info in graph_config['nodes'][slicing_point:]:
            node_inputs = [n[0] for n in node_info]
            for node_input in node_inputs:
                if node_input < slicing_point:
                    node_arg_dep.append(node_input)
        

        front_nodes = front_graph_json_data['nodes'][:slicing_point+1]
        front_arg_idx = 0
        while front_graph_json_data['arg_nodes'][front_arg_idx] < slicing_point:
            front_arg_idx += 1

        # Front setting
        front_arg_nodes = front_graph_json_data['arg_nodes'][:front_arg_idx]
        front_heads = [[slicing_point, 0, 0]]
        front_row_ptr = front_graph_json_data['node_row_ptr'][:slicing_point + 2]
        front_attr_dltype = front_graph_json_data['attrs']['dltype'][1][:slicing_point + 1]
        front_attr_shape = front_graph_json_data['attrs']['shape'][1][:slicing_point + 1]
        front_attr_storage_id = front_graph_json_data['attrs']['storage_id'][1][:slicing_point + 1]
        front_graph_json_data['nodes'] = front_nodes
        front_graph_json_data['arg_nodes'] = front_arg_nodes
        front_graph_json_data['heads'] = front_heads
        front_graph_json_data['node_row_ptr'] = front_row_ptr
        front_graph_json_data['attrs']['dltype'][1] = front_attr_dltype
        front_graph_json_data['attrs']['shape'][1] = front_attr_shape
        front_graph_json_data['attrs']['storage_id'][1] = front_attr_storage_id
        front_graph_config = json.dumps(front_graph_json_data)

        # Back               
