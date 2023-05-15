from train_and_test.data_process import test_data_read
from train_and_test.data_process import train_data_read

if __name__ == '__main__':
    # data_process test
    train_data_read(224)
    test_data_read(224)