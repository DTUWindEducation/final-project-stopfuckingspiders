import numpy as np

def calc_flow_angle(a, a_prime, V_0, omega, r):
    """
    phi [deg]:      Flow angle
    a [?]:          Axial induction factor
    a_prime [?]:    Tangential induction factor
    V_0 [m/s]:      Inflow wind speed
    omega [1/s]:    Rotational speed
    r [m]:          Span position
    """

    # Calculate the flow angle using the given formula
    # phi = arctan((1-a)*V_0 / ((1+a')*omega*r))
    numerator = (1 - a) * V_0
    denominator = (1 + a_prime) * omega * r
    phi = np.arctan(numerator / denominator)

    return phi

def calc_local_angle_of_attack(phi, theta_p, r):
    """
    alpha [deg]: local angle of attack
    phi [deg]: Flow angle
    theta_p [deg]: Blade_pitch_angle
    beta [deg]: local twist angle
    r [m]: Span position
    """

    beta = calc_twist_angle(r)
    alpha = phi  - (theta_p + beta)

    return alpha

# Local solidity (sigma)
def calc_local_solidity(r, B):

    c = lambda r: calc_chord_length(r) 
    Nominator = c * B
    Denominator = 2*np.pi*r

    sigma = Nominator/Denominator

    return sigma
# Calculate the tip speed ratio (lambda)
def calc_speed_ratio(rot_speed, inflow_wind_speed, rotor_radius):
    # lambda = omega * R / V
    speed_ratio = rot_speed * rotor_radius / inflow_wind_speed
    
    return  speed_ratio

def calc_chord_length(r):
    return 1

def calc_twist_angle(r):
    return 1