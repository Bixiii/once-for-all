import matplotlib.pyplot as plt
import numpy as np


class Visualisation:
    def __init__(self):
        self.mbv3_base_stage_width = [16, 24, 24, 24, 24, 40, 40, 40, 40, 80, 80, 80, 80, 112, 112, 112, 112, 160, 160,
                                      160, 160]

    def mbv3_barchart(self, subnet_config, save_path=None, title=None):
        layers = []
        for i in range(1, 21):

            if i <= subnet_config['d'][int((i - 1) / 4)] + int((i - 1) / 4) * 4:
                active = True
            else:
                active = False

            layers.append({'idx': i,
                           'active': active,
                           'e': subnet_config['e'][i - 1] * self.mbv3_base_stage_width[i - 1],
                           'ks': subnet_config['ks'][i - 1]})

        bar_idx = [layer['idx'] for layer in layers]
        bar_length = [0 if not layer['active'] else layer['e'] for layer in layers]
        bar_height = [0 if not layer['active'] else layer['ks'] / 8 for layer in layers]

        fig, ax = plt.subplots()
        plt.rcdefaults()

        chart = plt.barh(bar_idx, bar_length, align='edge', alpha=0.5, color='blue')

        for i, obj in enumerate(chart):
            obj.set_height(bar_height[i])

        ax.invert_yaxis()
        ax.set_yticks(bar_idx)
        ax.set_yticklabels(bar_idx)

        if title is not None:
            ax.set_title(title)
        ax.set_ylabel('layer index')
        ax.set_xlabel('number of channels')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if save_path is not None:
            plt.savefig(save_path)
        plt.show()


if __name__ == '__main__':
    test_graphics = [
        {
            'title': 'Default MobileNetV3 Large Architecture',
            'config': {'ks': [3, 3, 3, 3, 5, 5, 5, 3, 3, 3, 3, 3, 3, 5, 5, 5, 1, 7, 1, 1],
                       # TODO get right default config
                       'e': [3, 4, 6, 4, 6, 6, 4, 4, 4, 3, 3, 4, 6, 4, 6, 6, 4, 4, 4, 6],
                       'd': [2, 3, 4, 2, 3], 'r': 224},
            'save_path': 'figures/ofa_mbv3_default.png',
        },
        {
            'title': 'OFA Largest Network for MobileNetV3 Architecture',
            'config': {'ks': [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
                       'e': [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                       'd': [4, 4, 4, 4, 4], 'r': 224},
            'save_path': 'figures/ofa_mbv3_max.png',
        },
        {
            'title': 'OFA Subnet MobileNetV3 Architecture',
            'config': {'ks': [3, 5, 7, 3, 3, 3, 3, 3, 5, 5, 5, 5, 3, 3, 5, 3, 7, 5, 7, 5],
                       'e': [3, 4, 6, 4, 6, 6, 4, 4, 4, 3, 3, 4, 6, 4, 6, 6, 4, 4, 4, 6],
                       'd': [4, 2, 2, 3, 4], 'r': 192},
            'save_path': 'figures/ofa_mbv3_subnet.png',
        },
    ]

    drawing = Visualisation()
    drawing.mbv3_barchart(test_graphics[0]['config'], test_graphics[0]['save_path'], test_graphics[0]['title'])
