from rosenblatt import Rosenblatt_Perceptron

__author__ = 'Khashayar'
__email__ = 'khashayar@ghamati.com'


if __name__ == '__main__':

    rp = Rosenblatt_Perceptron(init_w=[0, 1], init_b=7)

    sample_data = [
        [10, 2, 1],
        [2, 30, 1],
        [46, 5, 1],
        [64, 7, 1],
        [8, 9, 1],
        [10, 11, 1],
        [12, 3, 1],
    ]

    for data in sample_data:
        rp.estimate_w_and_b(train_data=data)
        print(f'\nW is :\n {rp.get_w()}')

    rp.draw()
