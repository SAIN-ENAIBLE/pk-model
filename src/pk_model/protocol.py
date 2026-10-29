from time import perf_counter

import numpy as np

from .routes import validate_route
from .solve import solve_protocol

"""mg for the dose, ng for the amount, hours for time, L for volume, L/h for clearance and flow."""


class Protocol:
    def __init__(
        self, dose, central, compartments, t_span, t_eval=None, type="IV", Ka=None
    ):
        """
        Initialise the Protocol class.

        For SC, a scalar dose is an initial depot amount; a callable dose
        supplies an input rate to an initially empty depot. Ka is the
        first-order absorption rate in inverse-time units.
        """
        self.dose = dose
        self.central = central
        self.compartments = compartments
        self.compartment_count = len(compartments)
        self.compartment_names = list(compartments.keys())
        self.type = validate_route(type, Ka)
        self.Ka = Ka

        self.all_compartment_list = []
        self.qp_list = [
            values[f"q{i}"]
            for i, values in enumerate(self.compartments.values(), start=1)
        ]
        self.vp_list = [
            values[f"v{i}"]
            for i, values in enumerate(self.compartments.values(), start=1)
        ]

        self.Qp_list = [
            values[f"Q{i}"]
            for i, values in enumerate(self.compartments.values(), start=1)
        ]

        if t_eval is None:
            t_eval = np.linspace(t_span[0], t_span[1], 200)

        start_time = perf_counter()
        self.solution = solve_protocol(self, t_span, t_eval)
        self.solve_time = perf_counter() - start_time

        if not self.solution.success:
            raise RuntimeError(f"ODE solver failed: {self.solution.message}")

        self.t = self.solution.t
        self.y = self.solution.y

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
        self.all_compartment_list = [
            self.compartment_eq(Qp, qc, vc, qp, vp)
            for Qp, qp, vp in zip(Qp_list, qp_list, vp_list, strict=False)
        ]
        return self.all_compartment_list

    def central_eq(self, absorption, qc, vc, cl):
        return absorption - (qc / vc) * cl - sum(self.all_compartment_list)

    def system_eq(self, t, y):
        """
        Calculate the rate of change of drug quantity in the system.
        t: time
        y: list of drug quantities in each compartment (central and peripheral)
        return: list of rates of change of drug quantities in each compartment (dqc/dt, dqp/dt)
        """
        qc = y[0]  # the central compartment quantity
        qp_list = y[
            1 : 1 + self.compartment_count
        ]  # the peripheral compartment quantities

        vc = self.central["vc"]
        cl = self.central["cl"]

        dqp_list = self.all_compartment_eq(
            self.Qp_list,  # the fixed inter-compartmental clearances
            qc,
            vc,
            qp_list,  # the peripheral compartment quantities which get updated
            self.vp_list,  # the fixed compartment volumes
        )
        dose = self.dose(t) if callable(self.dose) else self.dose

        if self.type == "IV":
            dqc = self.central_eq(dose, qc, vc, cl)
            return (
                [dqc] + dqp_list
            )  # return the rate of change of drug quantity in the system (dqc/dt, dq1/dt, dq2/dt, ...)

        depot = y[-1]  # amount remaining at the injection site
        depot_input = dose if callable(self.dose) else 0.0
        d_depot = depot_input - self.Ka * depot
        absorption = self.Ka * depot
        dqc = self.central_eq(absorption, qc, vc, cl)
        return (
            [dqc] + dqp_list + [d_depot]
        )  # return the rate of change of drug quantity in the system (dqc/dt, dq1/dt, dq2/dt, ..., d_depot/dt)
