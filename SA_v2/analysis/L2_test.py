import sys
sys.path.append('..')
from utils import UTILS
from Hamiltonian import HAMILTONIAN
from model import MODEL
import pickle

def main():
    # Utility object for reading, writing parameters, etc. 
    utils = UTILS()

    # Reading parameters from para.dat file
    parameters = utils.read_parameter_file(file="../para.dat")
    parameters['root'] = "../data/"

    utils.read_command_line_arg(parameters,sys.argv)

    # Printing parameters for user
    utils.print_parameters(parameters)

    # Defining Hamiltonian
    H = HAMILTONIAN(**parameters)

    # Defines the model, and precomputes
    model = MODEL(H, parameters)

    # save interacting states
    if abs(parameters['J'] - 1.0) < 0.001 :
        with open('psi_L=2_J=1.pkl','wb') as f :
            pickle.dump([model.psi_i, model.psi_target], f)

    # load interacting states
    if abs(parameters['J']) < 0.0001:
        with open('psi_L=2_J=1.pkl','rb') as f :
            model.psi_i, model.psi_target = pickle.load(f)
    

    print(model.psi_i)



if __name__ == "__main__":
    main()




