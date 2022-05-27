import matplotlib.pyplot as plt


class Visualisation:
    def __init__(self):
        # variable layers
        self.mbv3_large_block_widths = [16, 24, 24, 24, 24, 40, 40, 40, 40, 80, 80, 80, 80, 112, 112, 112, 112, 160,
                                        160, 160]
        # can only me used for ANNETTE latency estimation
        self.mbv3_small_block_widths = [16, 24, 24, 24, 0, 40, 40, 40, 0, 48, 48, 48, 0, 96, 96, 96, 0, 0, 0, 0]

    def mbv3_barchart(self, subnet_config, save_path=None, title=None, show=False, show_fixed=True, relative=True):
        layers = []
        bar_idx = []
        bar_length = []
        bar_height = []
        bar_colour = []

        if show_fixed:
            bar_idx.append(1)
            bar_height.append(3 / 8)
            bar_colour.append('black')
            if relative:
                bar_length.append(100)
            else:
                bar_length.append(16)

            bar_idx.append(2)
            bar_height.append(3 / 8)
            bar_colour.append('black')
            if relative:
                bar_length.append(100)
            else:
                bar_length.append(16)

            idx = 2
        else:
            idx = 0

        for i in range(1, 21):
            if i <= subnet_config['d'][int((i - 1) / 4)] + int((i - 1) / 4) * 4:
                active = True
            else:
                active = False

            layers.append({'idx': idx + i,
                           'active': active,
                           'e': subnet_config['e'][i - 1] * self.mbv3_large_block_widths[i - 1],
                           'ks': subnet_config['ks'][i - 1]})

        bar_idx = bar_idx + [layer['idx'] for layer in layers]
        for idx in range(0, 20):  # 20 layers are dynamic
            if layers[idx]['active']:
                if relative:
                    bar_length.append(layers[idx]['e'] / (self.mbv3_large_block_widths[idx] * 6) * 100)
                else:
                    bar_length.append(layers[idx]['e'])
                bar_height.append(layers[idx]['ks'] / 8)
                bar_colour.append('blue')

            else:
                bar_length.append(0)
                bar_height.append(0)
                bar_colour.append('blue')

        if show_fixed:
            bar_idx.append(23)
            bar_height.append(1 / 8)
            bar_colour.append('black')
            if relative:
                bar_length.append(100)
            else:
                bar_length.append(960)
            bar_idx.append(24)
            bar_height.append(1 / 8)
            bar_colour.append('black')
            if relative:
                bar_length.append(100)
            else:
                bar_length.append(1280)

        fig, ax = plt.subplots()
        plt.rcdefaults()

        chart = plt.barh(bar_idx, bar_length, align='edge', alpha=0.5, color=bar_colour)

        for i, obj in enumerate(chart):
            obj.set_height(bar_height[i])

        ax.invert_yaxis()
        ax.set_yticks(bar_idx)
        ax.set_yticklabels(bar_idx)

        if title is not None:
            ax.set_title(title)
        ax.set_ylabel('Layer index')
        if relative:
            ax.set_xlabel('Number of channels (relative to max number channels) [%]')
        else:
            ax.set_xlabel('Number of channels')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if save_path is not None:
            plt.savefig(save_path)
        if show:
            plt.show()

        return fig


reference_networks = [
    {
        # 0 means that this block will be removed
        'title': 'Default MobileNetV3 Large Architecture',
        'config': {'ks': [3, 3, 3, 0, 0, 5, 5, 5, 0, 3, 3, 3, 3, 3, 3, 0, 0, 5, 5, 5],
                   'e':  [1, 2.67, 3, 0, 0, 1.8, 3, 3, 0, 3, 2.5, 2.3, 2.3, 4.29, 6, 0, 0, 4.2, 6, 6],
                   'd':  [2, 3, 4, 2, 3], 'r': 224},
        'save_path': 'figures/ofa_mbv3_default.png',
    },
    {
        'title': 'OFA Largest Network for MobileNetV3 Architecture',
        'config': {'ks': [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
                   'e':  [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                   'd':  [4, 4, 4, 4, 4], 'r': 224},
        'save_path': 'figures/ofa_mbv3_max.png',
    },
    {
        # TODO use different base width for this one
        'title': 'Default MobileNetV3 Small Architecture',
        'config': {'ks': [3, 3, 3, 0, 0, 5, 5, 5, 0, 5, 5, 0, 0, 5, 5, 5, 0, 0, 0, 0],
                   'e':  [1, 3, 3.67, 0, 0, 2.4, 6, 6, 0, 2.5, 3, 0, 0, 3, 6, 6, 0, 0, 0, 0],
                   'd':  [2, 3, 2, 3, 0], 'r': 224},
        'save_path': 'figures/ofa_mbv3_default.png',
    },
]

if __name__ == '__main__':

    drawing = Visualisation()
    drawing.mbv3_barchart(reference_networks[1]['config'], 'max_net.png', 'Largest OFA-MobileNetV3 Architecture', show=True,
                          show_fixed=True, relative=True)
