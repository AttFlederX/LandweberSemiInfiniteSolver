### Utility functions
def print_matrix(matrix):
    ''' Prints a rounded & formatted matrix into the console '''
    for i in matrix:
        for j in i:
            # print(j, end="\t")
            print('{:<05f}'.format(j), end='\t')
        print("\n")