
def get_einsum_groups_active(rst):
    einsums = rst["tl_rsts"].keys()
    assert len(einsums) != 0
    instances_1d = None
    instances_2d = None

    utils_1d = []
    utils_2d = []

    for einsum in einsums:
        e_util = max([rst["tl_rsts"][einsum]["tl_rst"]["comp_latency"],
                    rst["tl_rsts"][einsum]["tl_rst"]["mem_latency"]]) * \
                        rst["tl_rsts"][einsum]["tl_rst"]["mac_utilized_instances"]

        is_1d = "PE_col" not in rst["tl_rsts"][einsum]["factors"].keys()
        if is_1d:
            if instances_1d == None:
                instances_1d = rst["tl_rsts"][einsum]["tl_rst"]["mac_instances"]

            utils_1d.append(e_util)
        else:
            if instances_2d == None:
                instances_2d = rst["tl_rsts"][einsum]["tl_rst"]["mac_instances"]

            utils_2d.append(e_util)

    _rst = {}
    if instances_1d == None:
        _rst["1d"] = 0
    else:
        _rst["1d"] = sum(utils_1d) / instances_1d #(latency * instances_1d)
    if instances_2d == None:
        _rst["2d"] = 0
    else:
        _rst["2d"] = sum(utils_2d) / instances_2d #(latency * instances_2d)
    return _rst


def get_transfusion_active(rst):
    einsums = rst["tl_rsts"].keys()
    assert len(einsums) != 0
    instances_1d = None
    instances_2d = None

    utils_1d = []
    utils_2d = []

    active_latency = {}
    for einsum in einsums:
        e_util = max([rst["tl_rsts"][einsum]["comp_latency"],
                    rst["tl_rsts"][einsum]["mem_latency"]]) * \
                        rst["tl_rsts"][einsum]["mac_utilized_instances"]

        traffic = rst["tl_rsts"][einsum]["traffic_energy_foreach"]["traffic"]
        is_1d = traffic["reg_file_2d"]["read"] == 0 and \
            traffic["reg_file_2d"]["write"] == 0
        if is_1d:
            if instances_1d == None:
                instances_1d = rst["tl_rsts"][einsum]["mac_instances"]

            utils_1d.append(e_util)
        else:
            if instances_2d == None:
                instances_2d = rst["tl_rsts"][einsum]["mac_instances"]

            utils_2d.append(e_util)

    _rst = {}
    if instances_1d == None:
        _rst["1d"] = 0
    else:
        _rst["1d"] = sum(utils_1d) / instances_1d #(latency * instances_1d)
    if instances_2d == None:
        _rst["2d"] = 0
    else:
        _rst["2d"] = sum(utils_2d) / instances_2d #(latency * instances_2d)
    return _rst


# def get_active(arch_config: ArchConfig, model: str, seq_len: str):
#     ffn_fused = read_results("FFN", arch_config, model, seq_len, True)
#     ffn_unfused = read_results("FFN", arch_config, model, seq_len, False)
#     flat_fused = read_results("Flat", arch_config, model, seq_len, True)
#     flat_unfused = read_results("Flat", arch_config, model, seq_len, False)
#     fusemax_fused = read_results("FuseMax", arch_config, model, seq_len, True)
#     fusemax_unfused = read_results("FuseMax", arch_config, model, seq_len, False)
#     layer_norm_fused = read_results("LayerNorm", arch_config, model, seq_len, True)
#     layer_norm_unfused = read_results("LayerNorm", arch_config, model, seq_len, False)
#     qkv_fused = read_results("QKV", arch_config, model, seq_len, True)
#     qkv_unfused = read_results("QKV", arch_config, model, seq_len, False)
#     softmax_fused = read_results("Softmax", arch_config, model, seq_len, True)
#     softmax_unfused = read_results("Softmax", arch_config, model, seq_len, False)
#     unfused = read_results("Unfused", arch_config, model, seq_len)
#     transfusion = read_results("TransFusion", arch_config, model, seq_len)

#     #get_einsum_groups_utilization(ffn_fused)

#     flat_fused = {"latency": flat_fused["latency"] + softmax_fused["latency"],
#                   "energy": flat_fused["energy"] + softmax_fused["energy"],
#                   "tl_rsts": flat_fused["tl_rsts"] | softmax_fused["tl_rsts"]}
#     flat_unfused = {"latency": flat_unfused["latency"] + softmax_fused["latency"],
#                   "energy": flat_unfused["energy"] + softmax_fused["energy"],
#                   "tl_rsts": flat_unfused["tl_rsts"] | softmax_fused["tl_rsts"]}
#     unfused  = {"latency": unfused["latency"] + softmax_unfused["latency"],
#                 "energy": unfused["energy"] + softmax_unfused["energy"],
#                 "tl_rsts": unfused["tl_rsts"] | softmax_unfused["tl_rsts"]}


#     lat_func = lambda *rst: MODEL_FUNC[model](*[r["latency"] for r in rst])
#     eng_func = lambda *rst: MODEL_FUNC[model](*[r["energy"] for r in rst])

#     util_1d_func = lambda *rst: MODEL_FUNC[model](*[get_einsum_groups_active(r)["1d"] for r in rst])
#     util_2d_func = lambda *rst: MODEL_FUNC[model](*[get_einsum_groups_active(r)["2d"] for r in rst])
#     trans_util_1d_func = lambda *rst: MODEL_FUNC[model](*[get_transfusion_active(r)["1d"] for r in rst])
#     trans_util_2d_func = lambda *rst: MODEL_FUNC[model](*[get_transfusion_active(r)["2d"] for r in rst])
#     rst = {
#         "FuseMax+LayerFused": {
#             "latency": lat_func(qkv_fused, fusemax_fused, layer_norm_fused, ffn_fused),
#             "energy": eng_func(qkv_fused, fusemax_fused, layer_norm_fused, ffn_fused),
#             "1d": util_1d_func(qkv_fused, fusemax_fused, layer_norm_fused, ffn_fused),
#             "2d": util_2d_func(qkv_fused, fusemax_fused, layer_norm_fused, ffn_fused)
#         },
#         "FuseMax": {
#             "latency": lat_func(qkv_unfused, fusemax_unfused, layer_norm_unfused, ffn_unfused),
#             "energy": eng_func(qkv_unfused, fusemax_unfused, layer_norm_unfused, ffn_unfused),
#             "1d": util_1d_func(qkv_unfused, fusemax_unfused, layer_norm_unfused, ffn_unfused),
#             "2d": util_2d_func(qkv_unfused, fusemax_unfused, layer_norm_unfused, ffn_unfused)
#         },
#         "FLAT": {
#             "latency": lat_func(qkv_unfused, flat_unfused, layer_norm_unfused, ffn_unfused),
#             "energy": eng_func(qkv_unfused, flat_unfused, layer_norm_unfused, ffn_unfused),
#             "1d": util_1d_func(qkv_unfused, flat_unfused, layer_norm_unfused, ffn_unfused),
#             "2d": util_2d_func(qkv_unfused, flat_unfused, layer_norm_unfused, ffn_unfused)
#         },
#         "Unfused": {
#             "latency": lat_func(qkv_unfused, unfused, layer_norm_unfused, ffn_unfused),
#             "energy": eng_func(qkv_unfused, unfused, layer_norm_unfused, ffn_unfused),
#             "1d": util_1d_func(qkv_unfused, unfused, layer_norm_unfused, ffn_unfused),
#             "2d": util_2d_func(qkv_unfused, unfused, layer_norm_unfused, ffn_unfused)
#         },
#         "TransFusion": {
#             "latency": transfusion["latency"],
#             "energy": transfusion["energy"],
#             "1d": trans_util_1d_func(transfusion["einsums_outputs"]["QKV"], transfusion["einsums_outputs"]["MHA"], transfusion["einsums_outputs"]["LayerNorm"], transfusion["einsums_outputs"]["FFN"]),
#             "2d": trans_util_2d_func(transfusion["einsums_outputs"]["QKV"], transfusion["einsums_outputs"]["MHA"], transfusion["einsums_outputs"]["LayerNorm"], transfusion["einsums_outputs"]["FFN"])
#         }
#     }

#     for _, _rst in rst.items():
#         _rst["1d"] = _rst["1d"] / _rst["latency"]
#         _rst["2d"] = _rst["2d"] / _rst["latency"]

#     return rst



# print(get_active(ARCH_CLOUD, "BERT", "1K"))