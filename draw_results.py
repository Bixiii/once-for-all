from ofa.imagenet_classification.elastic_nn.networks import OFAMobileNetV3
import torch
from ofa.nas.efficiency_predictor import AnnetteLatencyModel
from ofa.tutorial import AccuracyPredictor, LatencyTable, FLOPsTable

mbv3_flops_config = [
    {
        'latency_constrain': 600,
        'accuracy': 0.849895477,
        'net_config': {'wid': None, 'ks': [3, 3, 3, 5, 3, 5, 7, 3, 7, 7, 3, 7, 7, 7, 7, 5, 3, 5, 5, 5],
                       'e': [4, 3, 6, 3, 6, 6, 4, 6, 4, 3, 6, 6, 3, 6, 6, 6, 6, 6, 6, 6], 'd': [4, 4, 4, 4, 4],
                       'r': [224], 'image_size': [224]}
    },
    {
        'latency_constrain': 550,
        'accuracy': 0.848451495,
        'net_config': {'wid': None, 'ks': [3, 5, 7, 5, 5, 3, 7, 5, 7, 7, 7, 7, 3, 5, 7, 7, 5, 5, 3, 7],
                       'e': [4, 3, 3, 3, 6, 4, 4, 6, 4, 6, 6, 6, 6, 6, 4, 4, 6, 6, 6, 4], 'd': [3, 4, 4, 4, 4],
                       'r': [224], 'image_size': [224]}
    },
    {
        'latency_constrain': 500,
        'accuracy': 0.848718464,
        'net_config': {'wid': None, 'ks': [5, 5, 3, 5, 5, 3, 3, 3, 7, 5, 3, 3, 3, 5, 7, 7, 5, 3, 5, 7],
                       'e': [4, 3, 3, 4, 6, 4, 4, 6, 4, 3, 4, 6, 6, 6, 4, 4, 6, 6, 6, 4], 'd': [3, 4, 4, 4, 4],
                       'r': [224], 'image_size': [224]}
    },
    {
        'latency_constrain': 450,
        'accuracy': 0.839696288,
        'net_config': {'wid': None, 'ks': [5, 5, 5, 7, 5, 5, 7, 7, 7, 5, 3, 5, 7, 3, 3, 3, 7, 3, 5, 7],
                       'e': [4, 4, 3, 4, 6, 4, 4, 3, 4, 6, 4, 6, 6, 6, 6, 3, 6, 6, 6, 4], 'd': [3, 4, 4, 4, 4],
                       'r': [208], 'image_size': [208]}

    },
    {
        'latency_constrain': 400,
        'accuracy': 0.838452995,
        'net_config': {'wid': None, 'ks': [3, 3, 3, 7, 7, 7, 7, 3, 7, 7, 7, 3, 7, 7, 3, 7, 3, 7, 3, 3],
                       'e': [4, 3, 3, 6, 6, 4, 6, 3, 4, 3, 6, 4, 6, 4, 4, 4, 6, 6, 6, 3], 'd': [2, 4, 4, 4, 4],
                       'r': [208], 'image_size': [208]}
    },
    {
        'latency_constrain': 350,
        'accuracy': 0.832341552,
        'net_config': {'wid': None, 'ks': [5, 5, 3, 3, 7, 5, 3, 3, 5, 5, 3, 5, 7, 7, 5, 5, 3, 3, 5, 7],
                       'e': [3, 3, 6, 3, 4, 3, 4, 4, 6, 6, 4, 3, 6, 4, 3, 3, 6, 6, 6, 4], 'd': [2, 3, 4, 4, 4],
                       'r': [208], 'image_size': [208]}
    },
    {
        'latency_constrain': 300,
        'accuracy': 0.822890043,
        'net_config': {'wid': None, 'ks': [3, 5, 5, 7, 7, 3, 5, 7, 7, 5, 7, 7, 5, 5, 7, 7, 5, 5, 5, 7],
                       'e': [3, 3, 3, 6, 6, 6, 4, 6, 6, 4, 3, 3, 4, 6, 6, 4, 6, 6, 6, 4], 'd': [3, 4, 4, 3, 4],
                       'r': [176], 'image_size': [176]}
    },
    {
        'latency_constrain': 250,
        'accuracy': 0.819211006,
        'net_config': {'wid': None, 'ks': [3, 3, 7, 5, 5, 5, 7, 3, 7, 7, 3, 3, 7, 7, 3, 7, 3, 5, 3, 3],
                       'e': [4, 3, 4, 3, 6, 6, 4, 3, 4, 3, 4, 3, 4, 4, 4, 3, 6, 6, 6, 6], 'd': [2, 4, 4, 3, 4],
                       'r': [176], 'image_size': [176]}
    },
    {
        'latency_constrain': 200,
        'accuracy': 0.806632996,
        'net_config': {'wid': None, 'ks': [5, 3, 3, 5, 7, 5, 3, 3, 7, 7, 5, 7, 5, 7, 3, 3, 3, 7, 7, 3],
                       'e': [3, 3, 4, 6, 4, 4, 4, 4, 4, 4, 6, 4, 4, 4, 4, 6, 6, 6, 4, 3], 'd': [2, 3, 2, 3, 4],
                       'r': [176], 'image_size': [176]}
    },
    {
        'latency_constrain': 150,
        'accuracy': 0.785192668,
        'net_config': {'wid': None, 'ks': [3, 3, 3, 3, 7, 3, 7, 3, 5, 3, 3, 3, 3, 3, 5, 3, 3, 3, 7, 7],
                       'e': [3, 3, 3, 4, 4, 3, 6, 6, 4, 3, 3, 4, 3, 4, 3, 4, 4, 3, 4, 3], 'd': [2, 2, 2, 3, 3],
                       'r': [176], 'image_size': [176]}
    },
    {
        'latency_constrain': 125,
        'accuracy': 0.770481229,
        'net_config': {'wid': None, 'ks': [3, 3, 5, 3, 5, 3, 3, 5, 5, 7, 7, 7, 7, 5, 5, 3, 3, 3, 3, 3],
                       'e': [3, 3, 6, 6, 3, 4, 6, 3, 4, 4, 4, 3, 4, 4, 4, 4, 6, 6, 4, 3], 'd': [2, 2, 2, 2, 2],
                       'r': [160], 'image_size': [160]}
    },
]

constraint_type = 'annette'
annette_models = ['dnndk-mixed',
                  'dnndk-ref_roofline',
                  'dnndk-roofline',
                  'dnndk-statistical',
                  'ncs2-mixed',
                  'ncs2-ref_roofline',
                  'ncs2-roofline',
                  'ncs2-statistical',
                  ]
annette_model = 'dnndk-mixed'

# accuracy predictor (pretrained from OFA repo)
cuda_available = torch.cuda.is_available()
accuracy_predictor = AccuracyPredictor(
    pretrained=True,
    device='cuda:0' if cuda_available else 'cpu'
)
# print(accuracy_predictor.model)

# OFA network
ks_list = [3, 5, 7]
depth_list = [2, 3, 4]
expand_list = [3, 4, 6]
num_classes = 1000

ofa_network = OFAMobileNetV3(
    n_classes=num_classes,
    dropout_rate=0,
    ks_list=ks_list,
    depth_list=depth_list,
    expand_ratio_list=expand_list,
)

# efficiency predictor
efficiency_predictor_flops = FLOPsTable(device='cuda:0' if cuda_available else 'cpu', batch_size=1)
efficiency_predictor_annette = AnnetteLatencyModel(ofa_network, model=annette_model)

# calculate given config
for config in mbv3_flops_config:
    efficiency_annette = efficiency_predictor_annette.predict_efficiency(config['net_config'])
    efficiency_flops = efficiency_predictor_flops.predict_efficiency(config['net_config'])
    accuracy = accuracy_predictor.predict_accuracy([config['net_config']])
    print('eff annette: ' + str(efficiency_annette) + ' eff flops: ' + str(efficiency_flops))
    print('acc: ' + str(accuracy[0][0]))

