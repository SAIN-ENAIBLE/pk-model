# Pharmacokinetic model

`pk-model` implements compartmental pharmacokinetic models consisting of a
central compartment and zero or more peripheral compartments.

Drug quantities are used as the model state variables.

## Central and peripheral compartments

Let:

- $q_c$ = drug quantity in the central compartment
- $V_c$ = central compartment volume
- $CL$ = central clearance
- $q_i$ = drug quantity in peripheral compartment $i$
- $V_i$ = volume of peripheral compartment $i$
- $Q_i$ = intercompartmental clearance between the central and peripheral compartment

The concentration in the central compartment is

$$
C_c = \frac{q_c}{V_c}.
$$

The concentration in peripheral compartment $i$ is

$$
C_i = \frac{q_i}{V_i}.
$$

Drug exchange between the central compartment and peripheral compartment $i$
is described by

$$
Q_i
\left(
\frac{q_c}{V_c}
-
\frac{q_i}{V_i}
\right).
$$

The central compartment therefore follows

$$
\frac{dq_c}{dt}
=
Dose(t)
-
CL\frac{q_c}{V_c}
-
\sum_i
Q_i
\left(
\frac{q_c}{V_c}
-
\frac{q_i}{V_i}
\right).
$$

Each peripheral compartment follows

$$
\frac{dq_i}{dt}
=
Q_i
\left(
\frac{q_c}{V_c}
-
\frac{q_i}{V_i}
\right).
$$

## Linear clearance

The current model supports linear elimination from the central compartment:

$$
R_{\mathrm{elim}} = CL\,C_c.
$$

The `Clear` class represents this clearance process.

## Intravenous administration

For an IV bolus, the administered dose is added directly to the central
compartment at the administration time.

Repeated IV boluses may be specified at different administration times.

## Subcutaneous administration

Subcutaneous administration is represented using an absorption depot.

For a bolus dose deposited in the SC compartment,

$$
\frac{dq_0}{dt} = -k_a q_0,
$$

where $q_0$ is the amount of drug in the depot and $k_a$ is the first-order
absorption rate constant.

Drug enters the central compartment at the rate

$$
R_{\mathrm{abs}} = k_a q_0.
$$

The central equation therefore receives $R_{\mathrm{abs}}$ as its drug input.

## Administration routes

The currently supported administration routes are:

- intravenous (`IV`)
- subcutaneous (`SC`)

Intramuscular (`IM`) and intradermal (`ID`) routes are recognised by the
software but are not currently implemented.