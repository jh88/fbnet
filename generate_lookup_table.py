from fbnet.blocks import get_super_net
from fbnet import lookup_table


def main():
    super_net = get_super_net(
        num_classes=10,
        bn=True,
        config={'ss': [1,1,2,2,1,1,1,1,1]}
    )

    table = lookup_table.get_lookup_table(
        super_net,
        inputs_shape=(1, 32, 32, 3),
        n=10
    )

    lookup_table.save(table, 'lookup_table.json')

if __name__ == '__main__':
    main()
