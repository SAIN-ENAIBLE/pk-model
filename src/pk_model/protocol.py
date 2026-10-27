#
# Protocol class
#


class IV:
    def __init__(self, dose, compartment_dict):
        self.dose = dose  # dose function
        self.compartment_dict = compartment_dict  # dictionary of compartment parameters
        self.central = compartment_dict['central']  # central compartment parameters
        

    def compartment_eq(self, Qp, qc, vc, qp, vp):
        """
        Calculate the rate of change of drug quantity in a compartment p.
        Qp: inter-compartmental clearance (rate of drug transfer between the centralcompartment and compartment p)
        qc: drug quantity in central compartment
        vc: volume of the central compartment
        qp: drug quantity in compartment p
        vp: volume of compartment p
        return: rate of change of drug quantity in compartment p (dqp/dt)
        """

        return Qp * (qc / vc - qp / vp)

    def all_compartment_eq(self, Qp_list, qc, vc, qp_list, vp_list):
        """
        Calculate the rate of change of drug quantity in all compartments.
        Qp_list: list of inter-compartmental clearances for each compartment
        qc: drug quantity in central compartment
        vc: volume of the central compartment
        qp_list: list of drug quantities in each compartment
        vp_list: list of volumes for each compartment
        return: list of rates of change of drug quantities in each compartment (dqp/dt)
        """
        self.all_compartment_list = [self.compartment_eq(Qp, qc, vc, qp, vp) for Qp, qp, vp in zip(Qp_list, qp_list, vp_list)]

    def central_eq(self, dose, qc, vc, cl):
        return dose - (qc / vc) * cl  - sum(self.all_compartment_list)
    

