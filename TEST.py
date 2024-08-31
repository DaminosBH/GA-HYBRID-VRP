import random
from gavrptw.core import run_gavrptw


def main():
    '''main()'''
    
    instance_name = 'C101'
    random.seed(35121711)
    ind_size = 100
    pop_size =10
    cx_pb = 0.8
    mut_pb = 0.1


    n_gen = 2000
    export_csv = True
  
    run_gavrptw(instance_name=instance_name, ind_size=ind_size, pop_size=pop_size, \
        cx_pb=cx_pb, mut_pb=mut_pb, n_gen=n_gen, export_csv=export_csv)


if __name__ == '__main__':
    main()
