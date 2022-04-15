import numpy as np

def main():
    K1 = np.asarray([9, 9, 9, 9, 9, 9, 4, 4, 3])
    K2 = np.asarray([5, 5, 5, 5, 5, 5, 6, 6, 6])
    K3 = np.asarray([5, 5, 5, 5, 5, 5, 6, 6, 6])
    K4 = np.asarray([5, 5, 5, 5, 5, 5, 6, 6, 6])
    M1 = np.asarray([32, 16, 16, 8, 8, 4, 8, 16, 16])
    M2 = np.asarray([128, 64, 64, 32, 32, 16, 32, 64, 128])
    M3 = np.asarray([128, 64, 64, 32, 32, 16, 32, 64, 128])
    M4 = np.asarray([128, 64, 64, 32, 32, 16, 32, 64, 128])
    C_in = np.asarray([2, 64, 64, 128, 128, 256, 256, 128, 64])
    C_in2 = np.asarray([1, 64, 64, 128, 128, 256, 256, 128, 64])
    C_out = np.asarray([64, 64, 128, 128, 256, 256, 128, 64, 32])

    # FLOP =  np.sum(K1 * K2 * K3 * M1 * M2 * M3 * C_in * C_out)
    # FLOP2 = np.sum(K1 * K2 * K3 * M1 * M2 * M3 * C_in2 * C_out)
    # # FLOP = np.sum(K1 * K2 * K3 *K4 * M1 * M2 * M3 * M4 * C_in * C_out)
    # print(FLOP + 2*FLOP2)
    # Param = np.sum(K1 * K2 * K3 * C_in * C_out)
    # Param2 = np.sum(K1 * K2 * K3 * C_in2 * C_out)
    # # Param = np.sum(K1 * K2 * K3 * K4 * C_in * C_out)
    # print(Param + 2*Param2)

    Fea = np.sum(M1 * M2 * M3 * C_out)
    # Fea = np.sum(M1 * M2 * M3 * M4 * C_out)
    # print(Fea)
    print(Fea*3)

if __name__ == '__main__':
    main()